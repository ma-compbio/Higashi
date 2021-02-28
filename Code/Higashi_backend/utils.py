import numpy as np
import torch
from tqdm import tqdm, trange
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score, pairwise_distances
from concurrent.futures import as_completed, ProcessPoolExecutor
# from pybloom_live import ScalableBloomFilter
from copy import deepcopy
from scipy.stats import pearsonr,spearmanr
import json

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
	vec = np.asarray(vec)
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