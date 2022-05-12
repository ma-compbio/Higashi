import numpy as np
import os, sys
import time
import h5py
import torch
import torch.nn.functional as F
import json
import pandas as pd
from scipy.sparse import diags, vstack
from scipy.stats import norm
from sklearn.preprocessing import normalize

from sklearn.metrics import pairwise_distances
torch.set_num_threads(1)

def moving_avg(adj, moving_range):
	adj_origin = adj.copy()
	adj = adj.copy()
	adj = adj * norm.pdf(0)
	for i in range(moving_range * 2):
		before_list = [adj_origin[0, :]] * (i+1) + [adj_origin[:-(i+1), :]]
		adj_before = vstack(before_list)
		after_list = [adj_origin[i+1:, :]] + [adj_origin[-1, :]] * (i+1)
		adj_after = vstack(after_list)
		adj = adj + (adj_after + adj_before) * norm.pdf((i+1) / moving_range)
	return adj


def skip_start_end(config, chrom="chr1"):
	res = config['resolution']
	if 'cytoband_path' in config:
		cytoband_path = config['cytoband_path']
		gap_tab = pd.read_table(config["cytoband_path"], sep="\t", header=None, comment='#')
		gap_tab.columns = ['chrom', 'start', 'end', 'sth', 'type']
		gap_list = gap_tab[(gap_tab["chrom"] == chrom) & (gap_tab["type"] == "acen")]
		start = np.floor((np.array(gap_list['start']) - 100000) / res).astype('int')
		end = np.ceil((np.array(gap_list['end']) + 100000) / res).astype('int')
	else:
		cytoband_path = None
		start = []
		end = []
	
	return start, end

def get_config(config_path = "./config.jSON"):
	c = open(config_path,"r")
	return json.load(c)

def generate_binpair( start, end, min_bin_, max_bin_, not_use_set=None):
	if not_use_set is None:
		not_use_set = set()
	samples = []
	for bin1 in range(start, end):
		if bin1 in not_use_set:
			continue
		for bin2 in range(bin1 + min_bin_, min(bin1 + max_bin_, end)):
			if bin2 in not_use_set:
				continue
			samples.append([bin1, bin2])
	samples = np.array(samples) + 1
	return samples

def get_weights(config):
	temp_dir = config['temp_dir']
	embedding_name = config['embedding_name']
	embed_dir = os.path.join(temp_dir, "embed")
	
	v = np.load(os.path.join(embed_dir, "%s_0_origin.npy" % embedding_name))
	distance = pairwise_distances(v, metric='euclidean')
	distance_sorted = np.sort(distance, axis=-1)
	distance /= np.quantile(distance_sorted[:, 1:15].reshape((-1)), q=0.25)

	new_w = np.exp(-distance)
	new_w /= np.sum(new_w, axis=-1)

	return new_w

def prep_one(weighted_adj, chrom_list, impute_list, local_transfer_range, cell):
	global cell_neighbor_list, weight_dict, sparse_chrom_list, cell_weight, origin_sparse_list
	cell_chrom_list, mtx_list = [], ["" for i in range(len(chrom_list))]
	
	if weighted_adj:
		for chrom_index_in_impute, chrom in enumerate(impute_list):
			c = chrom_list.index(chrom)
			mtx = 0
			for nbr_cell in cell_neighbor_list[cell]:
				balance_weight = weight_dict[(nbr_cell, cell)]
				mtx = mtx + balance_weight * sparse_chrom_list[c][nbr_cell - 1]
			
			adj = moving_avg(mtx, local_transfer_range)
			adj.data = np.log1p(adj.data)
			adj = normalize(adj, norm='l1', axis=1).astype('float32')
			Acoo = adj.tocoo()
			row_indices, column_indices, v = Acoo.row, Acoo.col, Acoo.data
			indice, v = torch.from_numpy(np.asarray([row_indices, column_indices])), torch.from_numpy(v)
			# a = origin_sparse_list[].toarray()
			# a = a / (np.sum(a)+1e-15) * a.shape[0]
			# mtx_list.append(a)
			cell_chrom_list.append([indice, v, adj.shape])
	else:
		# weight1 = cell_weight[cell - 1]
		# select = np.argsort(weight1)[::-1][:100]
		for chrom_index_in_impute, chrom in enumerate(impute_list):
			c = chrom_list.index(chrom)
			mtx = sparse_chrom_list[c][cell - 1]
			adj = moving_avg(mtx, local_transfer_range)
			adj.data = np.log1p(adj.data)
			adj = normalize(adj, norm='l1', axis=1).astype('float32')
			Acoo = adj.tocoo()
			row_indices, column_indices, v = Acoo.row, Acoo.col, Acoo.data
			indice, v = torch.from_numpy(np.asarray([row_indices, column_indices])), torch.from_numpy(v)
			# a = np.sum(origin_sparse_list[chrom_index_in_impute][select], axis=0).toarray()
			# a  = a + np.diag(np.sum(a, axis=-1) == 0)
			# a = normalize(a, axis=1, norm='l1')
			# a =  sqrt_norm(a)
			# a = origin_sparse_list[c][cell-1].toarray()
			# a = a / (np.sum(a) + 1e-15) * a.shape[0]
			# mtx_list[c] = a
			cell_chrom_list.append([indice, v, adj.shape])
			
	return cell, cell_chrom_list, mtx_list

def impute_process(config_path, model, name, mode, cell_start, cell_end, sparse_path, weighted_info=None):
	global cell_neighbor_list, weight_dict, sparse_chrom_list, cell_weight, origin_sparse_list
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	config = get_config(config_path)
	res = config['resolution']
	impute_list = config['impute_list']
	chrom_list = config['chrom_list']
	temp_dir = config['temp_dir']
	raw_dir = os.path.join(temp_dir, "raw")
	min_distance = config['minimum_distance']
	local_transfer_range = config['local_transfer_range']
	cell_weight = get_weights(config)
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
		
	# origin_sparse_list = []
	# for chrom in impute_list:
	# 	origin_sparse_list.append(np.load(os.path.join(raw_dir, "%s_sparse_adj.npy" % chrom), allow_pickle=True))
	
	model.eval()
	model.only_model = True
	embedding_init = model.encode1.static_nn
	# print ("start off hook")
	embedding_init.off_hook()
	# embedding_init.wstack = embedding_init.wstack.cpu()

	torch.cuda.empty_cache()
	
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
	route_nn_list = []
	for chrom in impute_list:
		j = chrom_list.index(chrom)
		route_nn_list.append(j+1)
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

	# bulk_adj = [embedding_init.embeddings[chrom+1].embedding.clone() for chrom in range(len(chrom_list))]
	
	model.encode1.dynamic_nn.forward = model.encode1.dynamic_nn.forward_off_hook
	with torch.no_grad():
		count = 0
		cell_list = list(np.arange(cell_start, cell_end))
		for i in cell_list:
			cell = i + 1
			cell, cell_chrom_list, mtx_list = prep_one(weighted_adj, chrom_list, impute_list, local_transfer_range, cell)
			
			# for chrom in range(len(chrom_list)):
			# 	a = mtx_list[chrom]
			# 	if len(a) == 0:
			# 		continue
			#
			# 	embedding_init.embeddings[chrom+1].embedding[:, :a.shape[-1]] = torch.from_numpy(a).to(device)
			# embedding_init = embedding_init.to(device)
			# embedding_init.off_hook()
			
			model.encode1.dynamic_nn.fix_cell2(cell, bin_ids, cell_chrom_list, local_transfer_range, route_nn_list=route_nn_list)
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
	embedding_init.on_hook()
	embedding_init.off_hook([0])
	model.train()
	model.only_model = False
	model.encode1.dynamic_nn.fix = False
	model.encode1.dynamic_nn.train()
	torch.cuda.empty_cache()
	for s in embedding_init.embeddings:
		s.embedding = s.embedding.to(device)
	# for t in embedding_init.targets:
	# 	t.embedding = t.embedding.to(device)
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
	print ("current_device", current_device)
	model = torch.load(path, map_location=current_device)
	print(config_path, path, name, mode, cell_start, cell_end)
	impute_process(config_path, model, name, mode, cell_start, cell_end, sparse_path, weighted_info)