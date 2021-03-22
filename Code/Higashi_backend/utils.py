import numpy as np
import torch
from tqdm import tqdm, trange
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score, pairwise_distances
from concurrent.futures import as_completed, ProcessPoolExecutor
from copy import deepcopy
from scipy.stats import pearsonr,spearmanr
import json
import os
import h5py
import pickle
from sklearn.linear_model import LinearRegression
import scipy
def get_config(config_path = "./config.jSON"):
	c = open(config_path,"r")
	return json.load(c)

def write_config(data, config_path):
	with open(config_path, 'w') as outfile:
		json.dump(data, outfile)

def transform_weight_class(weight, mean, neg_num):
	weight = np.log2(weight + 1)
	weight = weight / mean * neg_num
	return weight


def add_padding_idx(vec):
	if len(vec.shape) == 1:
		return np.asarray([np.sort(np.asarray(v) + 1).astype('int')
						 for v in tqdm(vec)])
	else:
		vec = np.asarray(vec) + 1
		vec = np.sort(vec, axis=-1)
		return vec.astype('int')


def np2tensor_hyper(vec, dtype):
	if len(vec.shape) == 1:
		return [torch.as_tensor(v, dtype=dtype) for v in vec]
	else:
		return torch.as_tensor(vec, dtype = dtype)

def pass_(x):
	return x

def roc_auc_cuda(y_true, y_pred):
	if not type(y_true) == np.ndarray:
		y_true = y_true.cpu().detach().numpy().reshape((-1, 1))
		y_pred = y_pred.cpu().detach().numpy().reshape((-1, 1))
	try:
		return roc_auc_score(
			y_true, y_pred), average_precision_score(
			y_true, y_pred)
	except BaseException:
		try:
			return pearsonr(y_true.reshape((-1)), y_pred.reshape((-1)))[0], spearmanr(y_true.reshape((-1)), y_pred.reshape((-1)))[0]
		except:
			return 0.0, 0.0


def accuracy(output, target):
	pred = output >= 0.5
	truth = target >= 0.5
	acc = torch.sum(pred.eq(truth))
	acc = float(acc) * 1.0 / (truth.shape[0] * 1.0)
	return acc



def get_neighbor(x, neighbor_mask):
	if len(neighbor_mask) == 1:
		result = set()
		result.add(tuple(x))
		return result
	temp = (x + neighbor_mask)
	temp = np.sort(temp, axis=-1)
	result = set()
	for t in temp:
		result.add(tuple(t))
	return result

def build_hash(data,compress,forward=True):
	if forward:
		func_ = pass_
	else:
		func_ = tqdm
			
	if compress:
		dict1 = ScalableBloomFilter(error_rate=1e-4,initial_capacity=len(data) * 10)
	else:
		dict1 = set()

	for datum in func_(data):
		# We need sort here to make sure the order is right
		datum.sort()
		dict1.add(tuple(datum))
	
		# nbs = get_neighbor(datum, neighbor_mask)
		# dict1.update(nbs)
			
	del data
	return dict1


def build_hash2(data):
	dict2 = set()
	for datum in tqdm(data):
		for x in datum:
			for y in datum:
				if x != y:
					dict2.add((x, y))
	return dict2


def build_hash3(data):
	dict2 = set()
	for datum in tqdm(data):
		for i in range(3):
			temp = np.copy(datum).astype('int')
			temp[i] = 0
			dict2.add(tuple(temp))

	return dict2


def parallel_build_hash(data, func, num, initial = None, compress = False):
	import multiprocessing
	cpu_num = multiprocessing.cpu_count()
	print ("dict building", data.shape)
	# data = np.array_split(data, cpu_num*5)
	dict1 = deepcopy(initial)
	pool = ProcessPoolExecutor(max_workers=cpu_num)
	process_list = []

	if func == 'build_hash':
		func = build_hash
	if func == 'build_hash2':
		func = build_hash2
	if func == 'build_hash3':
		func = build_hash3

	# for datum in data[:-1]:
	# 	process_list.append(pool.submit(func, datum, compress, False))
	# process_list.append(pool.submit(func, data, compress, False))
	#
	# for p in as_completed(process_list):
	# 	a = p.result()
	# 	if compress:
	# 		dict1 = dict1.union(a)
	# 	else:
	# 		dict1.update(a)
	#
	# pool.shutdown(wait=True)
	dict1 = build_hash(data, compress, False)

	return dict1

def generate_binpair( start, end, min_bin_, max_bin_):
	samples = []
	for bin1 in range(start, end):
		for bin2 in range(bin1 + min_bin_, min(bin1 + max_bin_, end)):
			samples.append([bin1, bin2])
	samples = np.array(samples) + 1
	return samples

def rankmatch(from_mtx, to_mtx):
	temp = np.sort(to_mtx.reshape((-1)))
	temp2 = from_mtx.reshape((-1))
	order = np.argsort(temp2)
	temp2[order] = temp
	return temp2.reshape((len(from_mtx), -1))


def linkhdf5_one_chrom(chrom, name, cell_id_splits, temp_dir, impute_list, name2=None):
	f = h5py.File(os.path.join(temp_dir, "%s_%s.hdf5" % (chrom, name)), "w")
	if name2 is not None:
		f1 = h5py.File(os.path.join(temp_dir, "%s_%s.hdf5" % (chrom, name2)), "r")
	bar = trange(len(np.concatenate(cell_id_splits)))
	for i, ids in enumerate(cell_id_splits):
		with h5py.File(os.path.join(temp_dir, "%s_%s_part_%d.hdf5" % (chrom, name, i)), "r") as input_f:
			if i == 0:
				f.create_dataset('coordinates', data=input_f['coordinates'])
			# print (input_f.keys(), "%s_%s_part_%d.hdf5" % (chrom, name, i))
			for cell in ids:
				v1 = np.array(input_f["cell_%d" % (cell)])
				if name2 is not None:
					v2 = np.array(f1["cell_%d" % (cell)])
					v = v1 / np.mean(v1) + v2 / np.mean(v2)
				else:
					v = v1 / np.mean(v1)
				f.create_dataset('cell_%d' % cell, data=v)
				bar.update(1)
				bar.refresh()
	f.close()
	if name2 is not None:
		f1.close()

def linkhdf5(name, cell_id_splits, temp_dir, impute_list, name2=None):
	print("start linking hdf5 files")
	pool = ProcessPoolExecutor(max_workers=len(impute_list))
	for chrom in tqdm(impute_list):
		pool.submit(linkhdf5_one_chrom, chrom, name, cell_id_splits, temp_dir, impute_list, name2)
	pool.shutdown(wait=True)
	# for chrom in impute_list:
	# 	for i in range(len(cell_id_splits)):
	# 		os.remove(os.path.join(temp_dir, "%s_%s_part_%d.hdf5" % (chrom, name, i)))

def modify_nbr_hdf5(name1, name2, temp_dir, impute_list, config):
	print ("Post processing step 1")
	neighbor = config['neighbor_num']
	for chrom in tqdm(impute_list):
		f1 = h5py.File(os.path.join(temp_dir, "%s_%s.hdf5" % (chrom, name1)), "r")
		f2 = h5py.File(os.path.join(temp_dir, "%s_%s.hdf5" % (chrom, name2)), "r+")
		
		for id_ in f1.keys():
			if "cell" in id_:
				data2 = f2[id_]
				v = (np.array(f2[id_]) + np.array(f1[id_])) / neighbor
				data2[...] = v
		f1.close()
		f2.close()

def rank_match_hdf5_one_chrom(name, temp_dir, chrom, config):
	f = h5py.File(os.path.join(temp_dir, "%s_%s.hdf5" % (chrom, name)), "r+")
	coordinates = np.array(f['coordinates']).astype('int')
	xs, ys = coordinates[:, 0], coordinates[:, 1]
	origin_sparse = np.load(os.path.join(temp_dir, "%s_sparse_adj.npy" % chrom), allow_pickle=True)
	
	bulk = np.array(np.sum(origin_sparse, axis=0).todense()) / len(origin_sparse)
	values = bulk[xs, ys]
	
	max_distance = config['maximum_distance']
	res = config['resolution']
	
	if max_distance > 0:
		max_bin = int(max_distance / res)
		
		length = len(values)
		nonzerosum = np.sum(values > 1e-15)
		final_values = [values[values > 1e-15]]
		
		if (np.sum(bulk > 1e-15) / length) > 1:
			k = max_bin + 1
			while nonzerosum < length:
				v = np.diag(bulk, k)
				final_values.append(v[v > 1e-15])
				nonzerosum += np.sum(v > 1e-15)
				k += 1
				
				if k == (origin_sparse[0].shape[0]):
					break
				# print (nonzerosum)
				
			final_values = np.concatenate(final_values, axis=0)
			if len(final_values) > length:
				values = final_values[:length]
			else:
				values = np.concatenate([final_values, final_values, np.zeros([length - len(final_values)])])[:length]
		else:
			values = np.sort(bulk.reshape((-1)))[::-1][:length]
	
	
	values_sorted = np.sort(values)
	print ("sorted", values_sorted, np.sum(values_sorted > 1e-15) / len(values_sorted), np.sum(bulk > 1e-15) / len(values_sorted))
	
	for id_ in trange(len(f.keys())-1):
		id_ = "cell_%d" % id_
		data = f[id_]
		v = np.array(f[id_])
		order = np.argsort(v)
		
		v[order] = values_sorted
		data[...] = v
	
	background = []
	random_choice = np.random.choice(np.arange(len(f.keys())-1), min(1000,len(f.keys())-1), replace=True)
	for id_ in random_choice:
		v = np.array(f["cell_%d" % id_])
		background.append(v)
	background = np.stack(background, axis=0)
	bg = np.quantile(background, 0.001, axis=0)
	del background
	for id_ in trange(len(f.keys())-1):
		id_ = "cell_%d" % id_
		data = f[id_]
		v = np.array(f[id_])
		v -= bg
		v[v < 0] = 0.0
		data[...] = v
	
	f.close()
	
	
	
def rank_match_hdf5(name, temp_dir, chrom_list, config):
	print("Post processing final step")
	pool = ProcessPoolExecutor(max_workers=len(chrom_list))
	for chrom in chrom_list:
		pool.submit(rank_match_hdf5_one_chrom, name, temp_dir, chrom, config)
	pool.shutdown(wait=True)

def get_neighbor(x, neighbor_mask):
	a = np.copy(x)
	temp = (a + neighbor_mask)
	temp = np.sort(temp, axis=-1)
	return list(temp)

def get_neighbor_mask():
	neighbor_mask = np.zeros((5, 3), dtype='int')
	count = 0
	for i in [-1, 0, 1]:
		for j in [-1, 0, 1]:
			if i !=0 and j !=0:
				continue
			neighbor_mask[count, 1] += i
			neighbor_mask[count, 2] += j
			count += 1
	return neighbor_mask

def remove_BE_linear(temp1, config, data_dir):
	if "batch_id" in config:
		if type(temp1) is list:
			temp1 = np.concatenate(temp1, axis=-1)

		batch_id_info = pickle.load(open(os.path.join(data_dir, "label_info.pickle"), "rb"))[config["batch_id"]]
		new_batch_id_info = np.zeros((len(batch_id_info), len(np.unique(batch_id_info))))
		for i, u in enumerate(np.unique(batch_id_info)):
			new_batch_id_info[batch_id_info == u, i] = 1
		batch_id_info = np.array(new_batch_id_info)
		temp1 = temp1 - LinearRegression().fit(batch_id_info, temp1).predict(batch_id_info)

	
	else:
		if type(temp1) is list:
			temp1 = np.concatenate(temp1, axis=-1)

	
	return temp1