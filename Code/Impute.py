import numpy as np
import os, sys
import time
import h5py
import torch
import torch.nn.functional as F
from Higashi_backend.utils import get_config, generate_binpair, skip_start_end

def impute_process(config_path, model, name, mode, cell_start, cell_end, sparse_path, weighted_info=None):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	config = get_config(config_path)
	res = config['resolution']
	impute_list = config['impute_list']
	chrom_list = config['chrom_list']
	temp_dir = config['temp_dir']
	min_distance = config['minimum_distance']
	local_transfer_range = config['local_transfer_range']
	if "impute_verbose" in config:
		impute_verbose = config['impute_verbose']
	else:
		impute_verbose = 10
		
	if min_distance < 0:
		min_bin = 0
	else:
		min_bin = int(min_distance / res)
	max_distance = config['maximum_distance']
	if max_distance < 0:
		max_bin = 1e5
	else:
		max_bin = int(max_distance / res)

	with h5py.File(os.path.join(temp_dir, "node_feats.hdf5"), "r") as input_f:
		num = np.array(input_f['num'])
	num_list = np.cumsum(num)
	
	sparse_chrom_list = np.load(sparse_path, allow_pickle=True)

	if weighted_info is not None:
		weighted_info = np.load(weighted_info, allow_pickle=True)
		
		
		

	model.eval()
	model.only_model = True
	embedding_init = model.encode1.static_nn
	# print ("start off hook")
	embedding_init.off_hook()
	embedding_init.wstack = embedding_init.wstack.cpu()

	torch.cuda.empty_cache()
	# print("off hook & save mem")
	
	with torch.no_grad():
		try:
			model.encode1.dynamic_nn.start_fix()
		except Exception as e:
			print("cannot start fix", e)
			raise EOFError
	
	chrom2info = {}
	big_samples = []
	bin_ids = []
	chrom_info = []
	big_samples_chrom = []
	slice_start = 0
	for chrom in impute_list:
		j = chrom_list.index(chrom)
		bin_ids.append(np.arange(num_list[j], num_list[j + 1]) + 1)
		chrom_info.append(np.ones(num_list[j + 1] - num_list[j]) * j)
		if "minimum_impute_distance" in config:
			min_bin_ = int(config["minimum_impute_distance"] / res)
		else:
			min_bin_ = min_bin
		
		if "maximum_impute_distance" in config:
			if config["maximum_impute_distance"] < 0:
				max_bin_ = int(1e5)
			else:
				max_bin_ = int(config["maximum_impute_distance"] / res)
		else:
			max_bin_ = max_bin

		skip_start, skip_end = skip_start_end(config, chrom)
		not_use_set = set()
		for s,e in zip(skip_start, skip_end):
			for bin_ in range(s, e):
				not_use_set.add(int(bin_) + num_list[j])

		samples_bins = generate_binpair( start=num_list[j], end=num_list[j + 1],
		                                min_bin_=min_bin_, max_bin_=max_bin_, not_use_set=not_use_set)
		
		samples = samples_bins - int(num_list[j]) - 1
		xs = samples[:, 0]
		ys = samples[:, 1]
		
		
		if mode == 'classification':
			activation = torch.sigmoid
		else:
			activation = F.softplus


		samples = np.concatenate([np.ones((len(samples_bins), 1), dtype='int'), samples_bins],
		                         axis=-1)
		
		f = h5py.File(os.path.join(temp_dir, "%s_%s.hdf5" % (chrom, name)), "w")
		to_save = np.stack([xs, ys], axis=-1)
			
		f.create_dataset("coordinates", data=to_save)
		big_samples.append(samples)
		big_samples_chrom.append(np.ones(len(samples), dtype='int') * j)
		chrom2info[chrom] = [slice_start, slice_start + len(samples), f]
		slice_start += len(samples)
	
	big_samples = np.concatenate(big_samples, axis=0)
	big_samples_chrom = np.concatenate(big_samples_chrom, axis=0)

	start = time.time()
	model.eval()
	print ("total number of triplets to predict:", big_samples.shape)
	
	if weighted_info is not None:
		weighted_adj = True
		cell_neighbor_list, weight_dict = weighted_info[0], weighted_info[1]
		
	else:
		weighted_adj = False

	# with h5py.File(os.path.join(temp_dir, "node_feats.hdf5"), "r") as input_f:
	# 	distance2weight = np.array(input_f['distance2weight']).reshape((-1, 1))
	# distance2weight = torch.from_numpy(distance2weight).to(device)

	with torch.no_grad():
		count = 0
		for i in range(cell_start, cell_end):
			cell = i + 1
			if weighted_adj:
				cell_chrom_list = []
				for chrom_index_in_impute, chrom in enumerate(impute_list):
					c = chrom_list.index(chrom)
					mtx = 0
					for nbr_cell in cell_neighbor_list[cell]:
						balance_weight = weight_dict[(nbr_cell, cell)]
						mtx = mtx + balance_weight * sparse_chrom_list[c][nbr_cell - 1]

					cell_chrom_list.append(mtx)
				cell_chrom_list = np.array(cell_chrom_list)
			else:
				cell_chrom_list = []
				for chrom_index_in_impute, chrom in enumerate(impute_list):
					c = chrom_list.index(chrom)
					mtx = sparse_chrom_list[c][cell-1]
					cell_chrom_list.append(mtx)
				cell_chrom_list = np.array(cell_chrom_list)

			try:
				model.encode1.dynamic_nn.fix_cell2(cell, bin_ids, cell_chrom_list, local_transfer_range)
			except:
				pass
			
			big_samples[:, 0] = cell
			proba = model.predict(big_samples, big_samples_chrom, verbose=False, batch_size=int(1e5),
			                      activation=activation, extra_info=None).reshape((-1))
			

			for chrom in impute_list:
				slice_start, slice_end, f = chrom2info[chrom]
				v = np.array(proba[slice_start:slice_end])
				f.create_dataset("cell_%d" % (cell-1), data=v, compression="gzip", compression_opts=6)

			count += 1
			if impute_verbose > 0:
				if (i-cell_start) % impute_verbose == 0:
					print("Imputing %s: %d of %d, takes %.2f s estimate %.2f s to finish" % (name, count, cell_end-cell_start,
																							 time.time() - start,
																							 (time.time() - start) / count * (cell_end - cell_start - count)))
			
	
	print("finish imputing, used %.2f s" % (time.time() - start))
	model.train()
	model.only_model = False
	model.encode1.dynamic_nn.fix = False
	torch.cuda.empty_cache()
	for s in embedding_init.embeddings:
		s.embedding = s.embedding.to(device)
	for t in embedding_init.targets:
		t.embedding = t.embedding.to(device)
	embedding_init.wstack = embedding_init.wstack.to(device)
	for chrom in chrom2info:
		slice_start, slice_end, f = chrom2info[chrom]
		f.close()

def get_free_gpu():
	os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > ./tmp')
	memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
	if len(memory_available) > 0:
		id = int(np.argmax(memory_available))
		print("setting to gpu:%d" % id)
		torch.cuda.set_device(id)
		return "cuda:%d" % id
	else:
		id =  np.random.choice(8, 1)
		print("setting to gpu:%d" % id)
		torch.cuda.set_device(int(id))
		return "cuda:%d" % id


if __name__ == '__main__':
	

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	config_path, path, name, mode, cell_start, cell_end, sparse_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
	
	weighted_info = sys.argv[8]
	if weighted_info == "None":
		weighted_info = None

	gpu_id = sys.argv[9]
	if gpu_id != "None":
		gpu_id = int(gpu_id)
		print("setting to gpu:%d" % gpu_id)
		torch.cuda.set_device(gpu_id)
		current_device = 'cuda:%d' % gpu_id
	else:
		if torch.cuda.is_available():
			current_device = get_free_gpu()
		else:
			current_device = 'cpu'
	model = torch.load(path, map_location=current_device)
	print(config_path, path, name, mode, cell_start, cell_end)
	impute_process(config_path, model, name, mode, cell_start, cell_end, sparse_path, weighted_info)