import math
import os
import pandas as pd
from Higashi_backend.utils import *
from Higashi_backend.Modules import *

import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from scipy.sparse import csr_matrix
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import h5py
import pickle
from sklearn.decomposition import PCA, TruncatedSVD, SparsePCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer
import subprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]


import argparse


def parse_args():
	parser = argparse.ArgumentParser(description="Higashi visualization tool")
	parser.add_argument('-c', '--config', type=str, default="./config.JSON")
	return parser.parse_args()

def get_free_gpu():
	os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > ./tmp')
	memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
	if len(memory_available) > 0:
		max_mem = np.max(memory_available)
		ids = np.where(memory_available == max_mem)[0]
		chosen_id = int(np.random.choice(ids, 1)[0])
		print("setting to gpu:%d" % chosen_id)
		torch.cuda.set_device(chosen_id)
	else:
		return
	
	
# Generate a indexing table of start and end id of each chromosome
def generate_chrom_start_end():
	print ("generating start/end dict for chromosome")
	chrom_size = pd.read_table(genome_reference_path, sep="\t", header=None)
	chrom_size.columns = ['chrom', 'size']
	# build a list that stores the start and end of each chromosome (unit of the number of bins)
	chrom_start_end = np.zeros((len(chrom_list), 2), dtype='int')
	for i, chrom in enumerate(chrom_list):
		size = chrom_size[chrom_size['chrom'] == chrom]
		size = size['size'][size.index[0]]
		n_bin = int(math.ceil(size / res))
		chrom_start_end[i, 1] = chrom_start_end[i, 0] + n_bin
		if i + 1 < len(chrom_list):
			chrom_start_end[i + 1, 0] = chrom_start_end[i, 1]
	
	print("chrom_start_end", chrom_start_end)
	np.save(os.path.join(temp_dir, "chrom_start_end.npy"), chrom_start_end)
	
# Extra the data.txt table
def extract_table():
	print ("extracting from data.txt")
	chrom_start_end = np.load(os.path.join(temp_dir, "chrom_start_end.npy"))
	data = pd.read_table(os.path.join(data_dir, "data.txt"), sep="\t")
	
	# ['cell_name','cell_id', 'chrom1', 'pos1', 'chrom2', 'pos2', 'count']
	print (data)
	pos1 = np.array(data['pos1'].values)
	pos2 = np.array(data['pos2'].values)
	bin1 = np.floor(pos1 / res).astype('int')
	bin2 = np.floor(pos2 / res).astype('int')
	
	new_bin1 = np.minimum(bin1, bin2)
	new_bin2 = np.maximum(bin1, bin2)
	bin1 = new_bin1
	bin2 = new_bin2
	chrom1, chrom2 = np.array(data['chrom1'].values), np.array(data['chrom2'].values)
	cell_id = np.array(data['cell_id'].values).astype('int')
	count = np.array(data['count'].values)
	
	new_chrom = np.ones_like(bin1, dtype='int') * -1
	for i, chrom in enumerate(chrom_list):
		mask = (chrom1 == chrom)
		new_chrom[mask] = i
		# Make the bin id for chromosome 2 starts at the end of the chromosome 1, etc...
		bin1[mask] += chrom_start_end[i, 0]
		bin2[mask] += chrom_start_end[i, 0]
	
	
	data = np.stack([cell_id, new_chrom, bin1, bin2], axis=-1)
	data = data[data[:, 1] >= 0]
	# print(data[(data[:, 0] == 0) & (data[:, 1] == 0)].shape, data[(data[:, 0] == 0) & (data[:, 1] == 0)])
	unique, inv, unique_counts = np.unique(data, axis=0, return_inverse=True, return_counts=True)
	new_count = np.zeros_like(unique_counts, dtype='float32')
	for i, iv in enumerate(tqdm(inv)):
		new_count[iv] += count[i]
	# print(new_count, unique_counts, data.shape)
	# print (unique[(unique[:, 0] == 0) & (unique[:, 1] == 0)].shape, unique[(unique[:, 0] == 0) & (unique[:, 1] == 0)])
	np.save(os.path.join(temp_dir, "data.npy"), unique, allow_pickle=True)
	np.save(os.path.join(temp_dir, "weight.npy"), new_count, allow_pickle=True)


def create_matrix_one_chrom(c, size, cell_size, temp, temp_weight, chrom_start_end, cell_num):
	cell_adj = []
	bin_adj = np.zeros((size, size))
	sparse_list = []
	# print (c, temp, temp_weight)
	bin_adj[temp[:, 2] - chrom_start_end[c, 0], temp[:, 3] - chrom_start_end[c, 0]] += temp_weight
	bin_adj = bin_adj + bin_adj.T
	
	for i in trange(cell_num):
		mask = temp[:, 0] == i
		temp2 = (temp[mask, 2:] - chrom_start_end[c, 0])
		temp2_scale = np.floor(temp2 / scale_factor).astype('int')
		temp_weight2 = temp_weight[mask]
		
		m1 = csr_matrix((temp_weight2, (temp2[:, 0], temp2[:, 1])), shape=(size, size))
		m1 = m1 + m1.T
		sparse_list.append(m1)
		
		m = np.zeros((cell_size, cell_size))
		for loc, w in zip(temp2_scale, temp_weight2):
			m[loc[0], loc[1]] += w
		
		m = m + m.T
		
		
		cell_adj.append(m.reshape((-1)))
	
	cell_adj = np.stack(cell_adj, axis=0)
	cell_adj = csr_matrix(cell_adj)
	print (cell_adj)
	bin_adj /= cell_num
	
	np.save(os.path.join(temp_dir, "%s_sparse_adj.npy" % chrom_list[c]), sparse_list)
	np.save(os.path.join(temp_dir, "%s_cell_adj.npy" % chrom_list[c]), cell_adj)
	np.save(os.path.join(temp_dir, "%s_bin_adj.npy" % chrom_list[c]), bin_adj)

# Generate matrices for feats and baseline
def create_matrix():
	print ("generating contact maps for baseline")
	data = np.load(os.path.join(temp_dir, "data.npy"))
	# print(data)
	weight = np.load(os.path.join(temp_dir, "weight.npy"))
	cell_num = np.max(data[:, 0]) + 1
	chrom_start_end = np.load(os.path.join(temp_dir, "chrom_start_end.npy"))
	# print("chrom_start_end", chrom_start_end)
	
	data_within_chrom_list = []
	weight_within_chrom_list = []
	pool = ProcessPoolExecutor(max_workers=cpu_num)
	p_list = []
	for c in range(len(chrom_list)):
		temp = data[data[:, 1] == c]
		temp_weight = weight[data[:, 1] == c]
		
		
		size = chrom_start_end[c, 1] - chrom_start_end[c, 0]
		cell_size = int(math.ceil(size / scale_factor))
		p_list.append(pool.submit(create_matrix_one_chrom, c, size, cell_size, temp, temp_weight, chrom_start_end, cell_num))
		
		data_within_chrom_list.append(temp)
		weight_within_chrom_list.append(temp_weight)
	pool.shutdown(wait=True)
	
	cell_feats = []
	for cell in trange(cell_num):
		mask = data[:, 0] == cell
		temp_weight = weight[mask]
		
		all_ = np.sum(temp_weight)
		cell_feats.append([all_])
	cell_feats = np.log10(np.array(cell_feats))
	cell_feats = StandardScaler().fit_transform(cell_feats)
	np.save(os.path.join(temp_dir, "cell_feats.npy"), cell_feats)
	
	
	data = np.concatenate(data_within_chrom_list)
	weight = np.concatenate(weight_within_chrom_list)
	
	# Get only the [cell, bin1, bin 2]
	data = data[:, [0, 2, 3]]
	# Make the bin id starts at max(cell id) + 1
	data[:, 1:] += np.max(data[:, 0]) + 1
	
	# generate the num vector that the main.py requires
	num = [np.max(data[:, 0]) + 1]
	for c in chrom_start_end:
		num.append(c[1] - c[0])
	# print(num)
	np.save(os.path.join(temp_dir, "num.npy"), num)
	
	num = [0] + list(num)
	num_list = np.cumsum(num)
	start_end_dict = np.zeros((num_list[-1], 2), dtype='int')
	id2chrom = np.zeros((num_list[-1] + 1), dtype='int')
	
	# Generate the start end dict that main.py requires
	# If maximum distance > 0, use the start end dict to restrict the sampled negative samples
	for i in range(len(num) - 1):
		start_end_dict[num_list[i]:num_list[i + 1], 0] = num_list[i]
		start_end_dict[num_list[i]:num_list[i + 1], 1] = num_list[i + 1]
		id2chrom[num_list[i] + 1:num_list[i + 1] + 1] = i - 1
		
	print("start end dict", start_end_dict)
	print("id2chrom", id2chrom)
	np.save(os.path.join(temp_dir, "start_end_dict.npy"), start_end_dict)
	np.save(os.path.join(temp_dir, "id2chrom.npy"), id2chrom)
	# print("data.shape", data.shape, data)
	weight = weight[data[:, 1] != data[:, 2]]
	data = data[data[:, 1] != data[:, 2]]
	
	# print("data.shape", data.shape)
	np.save(os.path.join(temp_dir, "filter_data.npy"), data)
	np.save(os.path.join(temp_dir, "filter_weight.npy"), weight)

# Code from scHiCluster
def neighbor_ave_gpu(A, pad):
	if pad == 0:
		return torch.from_numpy(A).float().to(device)
	ll = pad * 2 + 1
	conv_filter = torch.ones(1, 1, ll, ll).to(device)
	B = F.conv2d(torch.from_numpy(A[None, None, :, :]).float().to(device), conv_filter, padding=pad * 2)
	B = B[0, 0, pad:-pad, pad:-pad]
	return (B / float(ll * ll))

# Code from scHiCluster
def random_walk_gpu(A, rp, epochs=60):
	ngene, _ = A.shape
	A = A.float()
	A = A - torch.diag(torch.diag(A))
	A = A + torch.diag((torch.sum(A, 0) == 0).float())
	P = torch.div(A, torch.sum(A, 0))
	Q = torch.eye(ngene).to(device)
	I = torch.eye(ngene).to(device)
	for i in range(epochs):
		Q_new = (1 - rp) * I + rp * torch.mm(Q, P)
		delta = torch.norm(Q - Q_new, 2)
		Q = Q_new
		if delta < 1e-6:
			break
	return Q

# Code from scHiCluster
def impute_gpu(A):
	pad = 1
	rp = 0.5
	A = np.log2(A + 1)
	conv_A = neighbor_ave_gpu(A, pad)
	if rp == -1:
		Q2 = conv_A[:]
	else:
		Q2 = random_walk_gpu(conv_A, rp)
	return Q2.cpu().numpy()


def neighbor_ave_gpu2(A, pad):
	if pad == 0:
		return torch.from_numpy(A).float().to(device)
	ll = pad * 2 + 1
	conv_filter = torch.ones(1, 1, ll, ll).to(device)
	conv_filter[0, 0, 0, 0] = 0.25
	conv_filter[0, 0, 0, 1] = 0.5
	conv_filter[0, 0, 0, 2] = 0.25
	conv_filter[0, 0, 1, 0] = 0.5
	conv_filter[0, 0, 1, 1] = 1.0
	conv_filter[0, 0, 1, 2] = 0.5
	conv_filter[0, 0, 2, 0] = 0.25
	conv_filter[0, 0, 2, 1] = 0.5
	conv_filter[0, 0, 2, 2] = 0.25
	
	B = F.conv2d(torch.from_numpy(A[None, None, :, :]).float().to(device), conv_filter, padding=pad * 2)
	B = B[0, 0, pad:-pad, pad:-pad]
	return (B / float(ll * ll))


def conv_only(A):
	pad = 1
	conv_A = neighbor_ave_gpu2(A, pad)
	return conv_A.cpu().numpy()


# Run schicluster for comparisons
def impute_all():
	get_free_gpu()
	print("start conv random walk (scHiCluster) as baseline")
	for c in chrom_list:
		a = np.load(os.path.join(temp_dir, "%s_sparse_adj.npy"  % c), allow_pickle=True)
		# print("saving")
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
			
		# Minus one because the generate_binpair function would add one in the function
		samples = generate_binpair(0, a[0].shape[0], min_bin_, max_bin_) - 1
		with h5py.File(os.path.join(temp_dir, "rw_%s.hdf5" % c), "w") as f:
			f.create_dataset('coordinates', data = samples)
			for i, m in enumerate(tqdm(a)):
				m = impute_gpu(np.array(m.todense()))
				v = m[samples[:, 0], samples[:, 1]]
				
				f.create_dataset("cell_%d" % (i), data=v)

# Optional imputation for similar sparsity
def optional_impute_for_cell_adj():
	print ("optional impute")
	get_free_gpu()
	for c in chrom_list:
		a = np.load(os.path.join(temp_dir, "%s_cell_adj.npy" % c), allow_pickle=True).item()
		a = np.array(a.todense())
		
		impute_list = []
		sparsity = np.sum(a > 0 ,axis=-1) / a.shape[-1]
		upsampling2 = np.median(sparsity)
		
		
		for i, m in enumerate(tqdm(a)):
			b = m.reshape((int(np.sqrt(len(m))), -1))
			
			sparsity = np.sum(b > 0) / (b.shape[0] * b.shape[1])
			while sparsity < min(upsampling2 * 1.5, 1.0):
				# print ("sparsity", sparsity)
				b = conv_only(b)
				sparsity = np.sum(b > 0) / (b.shape[0] * b.shape[1])
			
			impute_list.append(b.reshape((-1)))
			
		impute_list = np.stack(impute_list, axis=0)
		print(impute_list.shape)
		thres = np.percentile(impute_list, 100 * (1 - upsampling2), axis=1)
		print(thres)
		impute_list = (impute_list > thres[:, None]).astype('float32') * impute_list
		print(impute_list)
		impute_list = csr_matrix(impute_list)
		sparsity = np.sum(impute_list > 0, axis=-1) / impute_list.shape[-1]
		print("sparsity", sparsity, np.median(sparsity), np.min(sparsity), np.max(sparsity))
		np.save(os.path.join(temp_dir, "%s_cell_impute.npy" % c), impute_list)

# Optional quantile normalization
def quantileNormalize(temp):
	temp = pd.DataFrame(temp.T)
	
	df = temp.copy()
	#compute rank
	dic = {}
	for col in df:
		dic.update({col : sorted(df[col])})
	sorted_df = pd.DataFrame(dic)
	rank = sorted_df.mean(axis = 1).tolist()
	#sort
	for col in df:
		t = np.searchsorted(np.sort(df[col]), df[col])
		df[col] = [rank[i] for i in t]
	return np.array(df).T

# generate feats for cell and bin nodes (one chromosome, multiprocessing)
def generate_feats_one(temp1,temp, total_embed_size, total_chrom_size, c):
	print (np.sum(temp1 > 0, axis=0))
	mask = np.array(np.sum(temp1 > 0, axis=0) > 0)
	mask = mask.reshape((-1))
	if type(temp) != np.ndarray:
		temp = np.array(temp.todense())
	size = int(total_embed_size / total_chrom_size * temp.shape[-1]) + 1
	temp = temp[:, mask]
	sparsity = np.sum(temp > 0 ,axis=-1) / temp.shape[-1]
	print ("sparsity", sparsity, np.median(sparsity), np.min(sparsity), np.max(sparsity))
	
	temp /= (np.sum(temp, axis=-1, keepdims=True) + 1e-15)
	if "optional_quantile" in config:
		if config['optional_quantile']:
			temp = quantileNormalize(temp)
	np.save(os.path.join(temp_dir, "%s_cell_feats.npy" % c), temp)

	temp1 = PCA(n_components=size).fit_transform(temp).astype('float32')
	print (temp.shape, temp1.shape)
	np.save(os.path.join(temp_dir, "%s_cell_PCA.npy" % c), temp1)

# generate feats for cell and bin nodes
def generate_feats(smooth_flag=False):
	print ("generating node attributes")
	pool = ProcessPoolExecutor(max_workers=cpu_num)
	chrom2adj = {}
	total_chrom_size = 0.0
	if smooth_flag:
		ext_str = 'impute'
	else:
		ext_str = 'adj'
	for c in chrom_list:
		try:
			temp = np.load(os.path.join(temp_dir, "%s_cell_%s.npy" % (c, ext_str)), allow_pickle=True).item()
		except:
			temp = np.load(os.path.join(temp_dir, "%s_cell_%s.npy" % (c, ext_str)), allow_pickle=True)
		
		chrom2adj[c] = temp
		total_chrom_size += list(temp.shape)[-1]
	
	if len(chrom_list) > 1:
		total_embed_size = int(temp.shape[0] * 1.3)
	else:
		total_embed_size = int(np.min(temp.shape) * 0.8)
	
	for c in chrom_list:
		temp = chrom2adj[c]
		pool.submit(generate_feats_one, temp, temp, total_embed_size, total_chrom_size, c)
	
	pool.shutdown(wait=True)


def process_signal_one(chrom):
	cmd = ["python", "Coassay_pretrain.py", args.config, chrom]
	subprocess.call(cmd)
	
def process_signal():
	print ("co-assay mode")
	signal_file = h5py.File(os.path.join(data_dir, "sc_signal.hdf5"), "r")
	signal_names = config["coassay_signal"]
	chrom2signals = {chrom:[] for chrom in chrom_list}
	for signal in signal_names:
		one_signal_stack = []
		signal_file_one = signal_file[signal]
		
		cells = np.arange(len(signal_file_one.keys()))
		for cell in cells:
			one_signal_stack.append(np.array(signal_file_one[str(cell)]))
		one_signal_stack = np.stack(one_signal_stack, axis=0).astype('float32')
		one_signal_stack = StandardScaler().fit_transform(one_signal_stack.reshape((-1, 1))).reshape(
			(len(one_signal_stack), -1))
		
		chrom_list_signal = np.array(signal_file[signal]["bin"]["chrom"])
		for chrom in chrom_list:
			chrom2signals[chrom].append(one_signal_stack[:, chrom_list_signal == chrom])
	
	for chrom in chrom_list:
		temp = chrom2signals[chrom]
		temp = np.concatenate(temp, axis=-1)
		np.save(os.path.join(temp_dir, "coassay_%s.npy" % chrom), temp)
	
	
	
	
	
	pool = ProcessPoolExecutor(max_workers=int(gpu_num * 2))
	for chrom in chrom_list:
		pool.submit(process_signal_one, chrom)
		time.sleep(3)
	
	pool.shutdown(wait=True)
	
	attributes_list = []
	for chrom in chrom_list:
		temp = np.load(os.path.join(temp_dir, "pretrain_coassay_%s.npy" % chrom))
		attributes_list.append(temp)
	attributes_list = np.concatenate(attributes_list)
	np.save(os.path.join(temp_dir, "pretrain_coassay.npy"), attributes_list)
	
args = parse_args()

config = get_config(args.config)
chrom_list = config['chrom_list']
res = config['resolution']
res_cell = config['resolution_cell']
bottle_neck = config['bottle_neck']
scale_factor = int(res_cell / res)
print ("scale_factor", scale_factor)


data_dir = config['data_dir']
temp_dir = config['temp_dir']
if not os.path.exists(temp_dir):
	os.mkdir(temp_dir)

cell_attr_dir = os.path.join(temp_dir, "cell_attributes")
bin_attr_dir = os.path.join(temp_dir, "bin_attributes")

if not os.path.exists(cell_attr_dir):
	os.mkdir(cell_attr_dir)

if not os.path.exists(bin_attr_dir):
	os.mkdir(bin_attr_dir)


contact_file_identifier = config['contact_file_identifier']
genome_reference_path = config['genome_reference_path']

if 'cpu_num' in config:
	cpu_num = config['cpu_num']
else:
	cpu_num = -1
	
if cpu_num < 0:
	cpu_num = multiprocessing.cpu_count()
	print("cpu_num", cpu_num)

gpu_num = config['gpu_num']


	
max_distance = config['maximum_distance']
if max_distance < 0:
	max_bin = 1e5
else:
	max_bin = int(max_distance / res)
print ("max bin", max_bin)


min_distance = config['minimum_distance']
if min_distance < 0:
	min_bin = 1e5
else:
	min_bin = int(min_distance / res)
print ("min bin", min_bin)


# generate_chrom_start_end()
# extract_table()
# create_matrix()
# impute_all()
optional_smooth_flag = False
if "optional_smooth" in config:
	if config['optional_smooth']:
		optional_smooth_flag = True
		optional_impute_for_cell_adj()
generate_feats(optional_smooth_flag)


if "coassay" in config:
	if config["coassay"]:
		process_signal()
