import numpy as np
import os, sys
import time
import h5py
import torch
import torch.nn.functional as F
from Higashi_backend.utils import get_config, generate_binpair
from tqdm import trange, tqdm
from scipy.sparse import csr_matrix

def impute_process(config_path, model, name, mode, cell_start, cell_end, sparse_path, weighted_info=None):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	config = get_config(config_path)
	sparse_chrom_list = np.load(sparse_path, allow_pickle=True)
	
	if weighted_info is not None:
		weighted_info = np.load(weighted_info, allow_pickle=True)
		
	res = config['resolution']
	impute_list = config['impute_list']
	chrom_list = config['chrom_list']
	temp_dir = config['temp_dir']
	min_distance = config['minimum_distance']
	if min_distance < 0:
		min_bin = 0
	else:
		min_bin = int(min_distance / res)
	max_distance = config['maximum_distance']
	if max_distance < 0:
		max_bin = 1e5
	else:
		max_bin = int(max_distance / res)
	
	num = np.load(os.path.join(temp_dir, "num.npy"))
	num_list = np.cumsum(num)
	
	model.eval()
	model.only_model = True
	embedding_init = model.encode1.static_nn
	
	embedding_init.off_hook()
	embedding_init.wstack = embedding_init.wstack.cpu()
	for s in embedding_init.embeddings:
		s.embedding = s.embedding.cpu()
	for t in embedding_init.targets:
		t.embedding = t.embedding.cpu()
	torch.cuda.empty_cache()
	print("off hook & save mem")
	
	with torch.no_grad():
		try:
			model.encode1.dynamic_nn.start_fix()
		except:
			print("cannot start fix")
			pass
	
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
		
		samples_bins = generate_binpair( start=num_list[j], end=num_list[j + 1],
		                                min_bin_=min_bin_, max_bin_=max_bin_)
		
		samples = samples_bins - int(num_list[j]) - 1
		xs = samples[:, 0]
		ys = samples[:, 1]
		
		
		if mode == 'classification':
			activation = torch.sigmoid
		elif mode == 'rank':
			activation = F.softplus
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
	# bin_ids = np.concatenate(bin_ids, axis=0)
	chrom_info = np.concatenate(chrom_info, axis=0).astype('int')
	start = time.time()
	model.eval()
	print (big_samples.shape)
	model.only_model = True
	
	if weighted_info is not None:
		weighted_adj = True
		cell_neighbor_list, weight_dict = weighted_info[0], weighted_info[1]
	else:
		weighted_adj = False
	
	
	if weighted_adj:
		print ("processing neighboring info")
		new_sparse_chrom_list = [[] for i in range(len(sparse_chrom_list))]
		for chrom in impute_list:
			c = chrom_list.index(chrom)
			new_cell_chrom_list = []
			for cell in np.arange(num_list[0])+1:
				mtx = 0
				for nbr_cell in cell_neighbor_list[cell]:
					balance_weight = weight_dict[(nbr_cell, cell)]
					mtx = mtx + balance_weight * sparse_chrom_list[c][nbr_cell - 1]

				new_cell_chrom_list.append(mtx)
			new_cell_chrom_list = np.array(new_cell_chrom_list)
			new_sparse_chrom_list[c] = new_cell_chrom_list
		new_sparse_chrom_list = np.array(new_sparse_chrom_list)
	else:
		new_sparse_chrom_list = sparse_chrom_list
	new_sparse_chrom_list = np.array(new_sparse_chrom_list)
	print (new_sparse_chrom_list.shape)
	with torch.no_grad():
		count = 0
		for i in range(cell_start, cell_end):
			cell = i + 1
			model.encode1.dynamic_nn.fix_cell2(cell, bin_ids, new_sparse_chrom_list[:, cell-1])
			big_samples[:, 0] = cell
			proba = model.predict(big_samples, big_samples_chrom, verbose=False, batch_size=int(4e5),
			                      activation=activation).reshape((-1))
			for chrom in impute_list:
				slice_start, slice_end, f = chrom2info[chrom]
				f.create_dataset("cell_%d" % (cell-1), data=proba[slice_start:slice_end])
			count += 1
			if (i-cell_start) % 10 == 0:
				print("Imputing %s: %d of %d, takes %.2f s" % (name, count, cell_end-cell_start, time.time() - start))
			
	
	print("finish writing, used %.2f s" % (time.time() - start))
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

def fetch_to_neighs(cell, nodes_chrom, bin_ids, part_sparse_chrom_list,  num_list):
	to_neighs = []
	for c, bin_ in zip(nodes_chrom, bin_ids):
		row = part_sparse_chrom_list[c][bin_ - 1 - int(num_list[c])]
		
		nbrs = row.nonzero()
		
		nbr_value = np.array(row.data).reshape((-1))
		nbrs = np.array(nbrs[1]).reshape((-1)) + 1 + int(num_list[c])
		if len(nbrs) > 0:
			temp = [nbrs, nbr_value]
		else:
			temp = []
		to_neighs.append(temp)
	to_neighs = np.array(to_neighs, dtype='object')
	return cell, to_neighs

def get_free_gpu():
	os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > ./tmp')
	memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
	if len(memory_available) > 0:
		id = int(np.argmax(memory_available))
		print("setting to gpu:%d" % id)
		torch.cuda.set_device(id)
		return "cuda:%d" % id
	else:
		return


if __name__ == '__main__':
	
	if torch.cuda.is_available():
		current_device = get_free_gpu()
	else:
		current_device = 'cpu'
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	config_path, path, name, mode, cell_start, cell_end, sparse_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
	if len(sys.argv) == 9:
		weighted_info = sys.argv[8]
	else:
		weighted_info = None
	model = torch.load(path, map_location=current_device)
	print(config_path, path, name, mode, cell_start, cell_end)
	impute_process(config_path, model, name, mode, cell_start, cell_end, sparse_path, weighted_info)