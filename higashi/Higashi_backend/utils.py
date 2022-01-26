import cooler
import numpy as np
import torch

try:
	get_ipython()
	from tqdm.notebook import tqdm, trange
except:
	from tqdm import tqdm, trange
	pass
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
import pandas as pd

tqdm.monitor_interval = 0

def get_config(config_path = "./config.jSON"):
	c = open(config_path,"r")
	return json.load(c)

def write_config(data, config_path):
	with open(config_path, 'w') as outfile:
		json.dump(data, outfile)


def fetch_batch_id(config, str1):
	batch_id_info = np.array(
		pickle.load(open(os.path.join(config["data_dir"], "label_info.pickle"), "rb"))[config[str1]])

	return batch_id_info

def transform_weight_class(weight, mean, neg_num):
	weight = np.log2(weight + 1)
	weight[weight >= np.quantile(weight, 0.99)] = np.quantile(weight, 0.99)
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
			y_true.reshape((-1)), y_pred.reshape((-1))), average_precision_score(
			y_true.reshape((-1)), y_pred.reshape((-1))), "auc", "aupr"
	except BaseException:
		try:
			return pearsonr(y_true.reshape((-1)), y_pred.reshape((-1)))[0], spearmanr(y_true.reshape((-1)), y_pred.reshape((-1)))[0], "pearson", "spearman"
		except:
			return 0.0, 0.0, "error", "error"


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

def skip_start_end(config, chrom="chr1"):
	res = config['resolution']
	if 'cytoband_path' in config:
		cytoband_path = config['cytoband_path']
		gap_tab = pd.read_table(config["cytoband_path"], sep="\t", header=None)
		gap_tab.columns = ['chrom', 'start', 'end', 'sth', 'type']
		gap_list = gap_tab[(gap_tab["chrom"] == chrom) & (gap_tab["type"] == "acen")]
		start = np.floor((np.array(gap_list['start']) - 100000) / res).astype('int')
		end = np.ceil((np.array(gap_list['end']) + 100000) / res).astype('int')
	else:
		cytoband_path = None
		start = []
		end = []
		
	
	return start, end

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
		ids = np.copy(ids)
		with h5py.File(os.path.join(temp_dir, "%s_%s_part_%d.hdf5" % (chrom, name, i)), "r") as input_f:
			if i == 0:
				f.create_dataset('coordinates', data=input_f['coordinates'])
			for cell in ids:
				try:
					v1 = np.array(input_f["cell_%d" % (cell)])
					if name2 is not None:
						v2 = np.array(f1["cell_%d" % (cell)])
						v = v1 / np.mean(v1) + v2 / np.mean(v2)
					else:
						v = v1 / np.mean(v1)
					f.create_dataset('cell_%d' % cell, data=v, compression="gzip", compression_opts=6)
				except:
					pass
				bar.update(1)

	f.close()
	if name2 is not None:
		f1.close()
	print ("start removing temp files")
	for i in range(len(cell_id_splits)):
		os.remove(os.path.join(temp_dir, "%s_%s_part_%d.hdf5" % (chrom, name, i)))

def linkhdf5(name, cell_id_splits, temp_dir, impute_list, name2=None):
	print("start linking hdf5 files")
	pool = ProcessPoolExecutor(max_workers=3)
	for chrom in tqdm(impute_list):
		# linkhdf5_one_chrom( chrom, name, np.copy(cell_id_splits), temp_dir, impute_list, name2)
		pool.submit(linkhdf5_one_chrom, chrom, name, np.copy(cell_id_splits), temp_dir, impute_list, name2)
	pool.shutdown(wait=True)
	# for chrom in impute_list:
	# 	for i in range(len(cell_id_splits)):
	# 		os.remove(os.path.join(temp_dir, "%s_%s_part_%d.hdf5" % (chrom, name, i)))



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

def remove_BE_linear(temp1, config, data_dir, cell_feats1):
	
	if "batch_id" in config:
		print ("initial removing BE")
		if type(temp1) is list:
			temp1 = np.concatenate(temp1, axis=-1)
		cell_feats1 = cell_feats1.detach().cpu().numpy()
		temp1 = temp1 - LinearRegression().fit(cell_feats1, temp1).predict(cell_feats1)
		# batch_id_info = np.array(pickle.load(open(os.path.join(data_dir, "label_info.pickle"), "rb"))[config["batch_id"]]).reshape((-1))
		# new_batch_id_info = np.zeros((len(batch_id_info), len(np.unique(batch_id_info))))
		# for i, u in enumerate(np.unique(batch_id_info)):
		# 	new_batch_id_info[batch_id_info == u, i] += 1
		# batch_id_info = np.array(new_batch_id_info)
		# temp1 = temp1 - LinearRegression().fit(batch_id_info, temp1).predict(batch_id_info)
	elif "regress_cov" in config:
		if config['regress_cov']:
			print("initial removing BE")
			if type(temp1) is list:
				temp1 = np.concatenate(temp1, axis=-1)
			cell_feats1 = cell_feats1.detach().cpu().numpy()
			temp1 = temp1 - LinearRegression().fit(cell_feats1, temp1).predict(cell_feats1)
		else:
			if type(temp1) is list:
				temp1 = np.concatenate(temp1, axis=-1)
	else:
		# print("initial removing BE")
		if type(temp1) is list:
			temp1 = np.concatenate(temp1, axis=-1)
		# cell_feats1 = cell_feats1.detach().cpu().numpy()
		# temp1 = temp1 - LinearRegression().fit(cell_feats1, temp1).predict(cell_feats1)
	return temp1