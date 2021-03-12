import math
import os
import pandas as pd
from Higashi_backend.utils import *
from Higashi_backend.Modules import *
from Higashi_analysis.Higashi_analysis import *
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from scipy.sparse import csr_matrix, vstack
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import h5py
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer
import subprocess
from fbpca import pca

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]


import argparse


def parse_args():
	parser = argparse.ArgumentParser(description="Higashi Processing")
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
	
	
def data2triplets(data, chrom_start_end, verbose):
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
	
	del data
	
	new_chrom = np.ones_like(bin1, dtype='int') * -1
	for i, chrom in enumerate(chrom_list):
		mask = (chrom1 == chrom)
		new_chrom[mask] = i
		# Make the bin id for chromosome 2 starts at the end of the chromosome 1, etc...
		bin1[mask] += chrom_start_end[i, 0]
		bin2[mask] += chrom_start_end[i, 0]
	
	data = np.stack([cell_id, new_chrom, bin1, bin2], axis=-1)
	count = count[data[:, 1] >= 0]
	data = data[data[:, 1] >= 0]
	
	unique, inv, unique_counts = np.unique(data, axis=0, return_inverse=True, return_counts=True)
	new_count = np.zeros_like(unique_counts, dtype='float32')
	func1 = tqdm if verbose else pass_
	for i, iv in enumerate(func1(inv)):
		new_count[iv] += count[i]
		
	return unique, new_count

# Extra the data.txt table
# Memory consumption re-optimize
def extract_table():
	print ("extracting from data.txt")
	chrom_start_end = np.load(os.path.join(temp_dir, "chrom_start_end.npy"))
	
	if "structured" in config:
		if config["structured"]:
			chunksize = int(5e5)
			unique, new_count = [], []
			cell_tab = []
			line_count = 0
			
			p_list = []
			pool = ProcessPoolExecutor(max_workers=cpu_num)
			print ("First calculating how many lines are there")
			with open(os.path.join(data_dir, "data.txt"), 'r') as csv_file:
				for line in csv_file:
					line_count += 1
			print("There are %d lines" % line_count)
			bar = trange(line_count, desc=' - Processing ', leave=False, )
			with open(os.path.join(data_dir, "data.txt"), 'r') as csv_file:
				chunk_count = 0
				reader =  pd.read_csv(csv_file, chunksize=chunksize, sep="\t")
				for chunk in reader:
					# print (chunk)
					if len(chunk['cell_id'].unique()) == 1:
						# Only one cell, keep appending
						cell_tab.append(chunk)
					else:
						# More than one cell, append all but the last part
						last_cell = np.array(chunk.tail(1)['cell_id'])[0]
						# print (last_cell)
						tails = chunk.iloc[np.array(chunk['cell_id']) != last_cell, :]
						head = chunk.iloc[np.array(chunk['cell_id']) == last_cell, :]
						cell_tab.append(tails)
						cell_tab = pd.concat(cell_tab, axis=0)
						p_list.append(pool.submit(data2triplets, cell_tab, chrom_start_end, False))
						cell_tab = [head]
					chunk_count += 1
					if chunk_count > cpu_num * 2:
						for p in as_completed(p_list):
							u_, n_ = p.result()
							unique.append(u_)
							new_count.append(n_)
							
							bar.update(n=chunksize)
						chunk_count = 0
						p_list = []
						
			for p in as_completed(p_list):
				u_, n_ = p.result()
				unique.append(u_)
				new_count.append(n_)
				
				bar.update(n=chunksize)
			unique, new_count = np.concatenate(unique, axis=0), np.concatenate(new_count, axis=0)
		else:
			data = pd.read_table(os.path.join(data_dir, "data.txt"), sep="\t")
			
			# ['cell_name','cell_id', 'chrom1', 'pos1', 'chrom2', 'pos2', 'count']
			print(data)
			unique, new_count = data2triplets(data, chrom_start_end, verbose=True)
	else:
		data = pd.read_table(os.path.join(data_dir, "data.txt"), sep="\t")
	
		# ['cell_name','cell_id', 'chrom1', 'pos1', 'chrom2', 'pos2', 'count']
		print (data)
		unique, new_count = data2triplets(data, chrom_start_end, verbose=True)
		
	np.save(os.path.join(temp_dir, "data.npy"), unique, allow_pickle=True)
	np.save(os.path.join(temp_dir, "weight.npy"), new_count, allow_pickle=True)


def create_matrix_one_chrom(c, size, cell_size, temp, temp_weight, chrom_start_end, cell_num):
	cell_adj = []
	bin_adj = np.zeros((size, size))
	sparse_list = []
	for i in range(len(temp)):
		bin_adj[temp[i, 2] - chrom_start_end[c, 0], temp[i, 3] - chrom_start_end[c, 0]] += temp_weight[i]
	bin_adj = bin_adj + bin_adj.T
	read_count = []
	
	sparsity_metric = []
	
	for i in trange(cell_num):
		mask = temp[:, 0] == i
		temp2 = (temp[mask, 2:] - chrom_start_end[c, 0])
		temp2_scale = np.floor(temp2 / scale_factor).astype('int')
		temp_weight2 = temp_weight[mask]
		
		read_count.append(np.sum(temp_weight2))
		m1 = csr_matrix((temp_weight2, (temp2[:, 0], temp2[:, 1])), shape=(size, size))
		m1 = m1 + m1.T
		sparse_list.append(m1)
		
		# m = np.zeros((cell_size, cell_size))
		# for loc, w in zip(temp2_scale, temp_weight2):
		# 	m[loc[0], loc[1]] += w
		m = csr_matrix((temp_weight2, (temp2_scale[:, 0], temp2_scale[:, 1])), shape=(cell_size, cell_size))
		m = m + m.T

		m = m / (np.sum(m) + 1e-15)
		cell_adj.append(m.reshape((1, -1)))
		
		if res_cell != 1000000:
			scale_factor2 = int(1000000 / res)
			size_metric = int(math.ceil(size * res / 1000000))
			temp2_scale = np.floor(temp2 / scale_factor2).astype('int')
			m1 = csr_matrix((temp_weight2, (temp2_scale[:, 0], temp2_scale[:, 1])), shape=(size_metric, size_metric))
			m1 = m1 + m1.T
			sparsity_metric.append(m1.reshape((1, -1)))
			
	# cell_adj = np.stack(cell_adj, axis=0)
	# cell_adj = csr_matrix(cell_adj)
	cell_adj = vstack(cell_adj).tocsr()
	bin_adj /= cell_num
	if res_cell != 1000000:
		# sparsity_metric = np.stack(sparsity_metric, axis=0)
		# sparsity_metric = csr_matrix(sparsity_metric)
		sparsity_metric = vstack(sparsity_metric).tocsr()
		np.save(os.path.join(temp_dir, "%s_sparsity_metric_adj.npy" % chrom_list[c]), sparsity_metric)
	np.save(os.path.join(temp_dir, "%s_sparse_adj.npy" % chrom_list[c]), sparse_list)
	
	new_temp = []
	for t in tqdm(sparse_list):
		t.setdiag(0)
		t.eliminate_zeros()
		new_temp.append(t)
	
	np.save(os.path.join(temp_dir, "%s_cell_adj.npy" % chrom_list[c]), cell_adj)
	np.save(os.path.join(temp_dir, "%s_bin_adj.npy" % chrom_list[c]), bin_adj)
	return read_count, np.array(new_temp), c

# Generate matrices for feats and baseline
def create_matrix():
	print ("generating contact maps for baseline")
	data = np.load(os.path.join(temp_dir, "data.npy"))
	weight = np.load(os.path.join(temp_dir, "weight.npy"))
	
	
	
	cell_num = np.max(data[:, 0]) + 1
	chrom_start_end = np.load(os.path.join(temp_dir, "chrom_start_end.npy"))
	
	data_within_chrom_list = []
	weight_within_chrom_list = []
	pool = ProcessPoolExecutor(max_workers=10)
	p_list = []
	
	cell_feats = [[] for i in range(len(chrom_list))]
	sparse_chrom_list = [[] for i in range(len(chrom_list))]
	
	save_mem = True if len(data) > 1e8 else False
	print (len(data), save_mem)
	for c in range(len(chrom_list)):
		temp = data[data[:, 1] == c]
		temp_weight = weight[data[:, 1] == c]
		
		
		size = chrom_start_end[c, 1] - chrom_start_end[c, 0]
		cell_size = int(math.ceil(size / scale_factor))
		
		data_within_chrom_list.append(temp)
		weight_within_chrom_list.append(temp_weight)
		
		if save_mem:
			chrom_count, non_diag_sparse, c = create_matrix_one_chrom( c, size, cell_size, temp, temp_weight, chrom_start_end, cell_num)
			cell_feats[c] = chrom_count
			sparse_chrom_list[c] = non_diag_sparse
		else:
			p_list.append(
				pool.submit(create_matrix_one_chrom, c, size, cell_size, temp, temp_weight, chrom_start_end, cell_num))
			
	if not save_mem:
		for p in as_completed(p_list):
			chrom_count, non_diag_sparse, c = p.result()
			cell_feats[c] = chrom_count
			sparse_chrom_list[c] = non_diag_sparse
	sparse_chrom_list = np.array(sparse_chrom_list)
	np.save(os.path.join(temp_dir, "sparse_nondiag_adj_nbr_1.npy"), sparse_chrom_list)
	cell_feats = np.stack(cell_feats, axis=-1)
	pool.shutdown(wait=True)
	cell_feats = np.log10(np.array(cell_feats) + 1)
	cell_feats = StandardScaler().fit_transform(cell_feats)
	np.save(os.path.join(temp_dir, "cell_feats.npy"), cell_feats)
	
	
	data = np.concatenate(data_within_chrom_list)
	weight = np.concatenate(weight_within_chrom_list)
	
	# Get only the [cell, bin1, bin 2]
	chrom_info = data[:, 1]
	data = data[:, [0, 2, 3]]
	# Make the bin id starts at max(cell id) + 1
	data[:, 1:] += np.max(data[:, 0]) + 1
	
	# generate the num vector that the main.py requires
	num = [np.max(data[:, 0]) + 1]
	for c in chrom_start_end:
		num.append(c[1] - c[0])
		
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
		

	np.save(os.path.join(temp_dir, "start_end_dict.npy"), start_end_dict)
	np.save(os.path.join(temp_dir, "id2chrom.npy"), id2chrom)
	mask = data[:, 1] != data[:, 2]
	weight = weight[mask]
	data = data[mask]
	chrom_info = chrom_info[mask]
	
	# Add 1 for padding idx
	np.save(os.path.join(temp_dir, "filter_data.npy"), data + 1)
	np.save(os.path.join(temp_dir, "filter_chrom.npy"), chrom_info)
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
	Q = Q - torch.diag(torch.diag(Q))
	return Q

# Code from scHiCluster
def impute_gpu(A):
	pad = 1
	rp = 0.5
	conv_A = neighbor_ave_gpu(A, pad)
	if rp == -1:
		Q2 = conv_A[:]
	else:
		Q2 = random_walk_gpu(conv_A, rp)
	return Q2.cpu().numpy()

def schicluster(chrom_matrix_list, dim, prct=20):
	matrix = []
	print ("start schiclsuter")
	for chrom in chrom_matrix_list:
		print (chrom.shape)
		chrom = chrom.reshape(len(chrom), -1)
		thres = np.percentile(chrom, 100 - prct, axis=1)
		chrom_bin = (chrom > thres[:, None]).astype('float32')
		ndim = int(min(chrom_bin.shape) * 0.2) - 1
		pca = PCA(n_components=ndim)
		R_reduce = pca.fit_transform(chrom_bin)
		
		print (chrom.shape, R_reduce.shape)
		matrix.append(R_reduce)
	matrix = np.concatenate(matrix, axis=1)
	print ("concatenate", matrix.shape)
	pca = PCA(n_components=dim)
	matrix_reduce = pca.fit_transform(matrix)
	return matrix_reduce

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
	
# generate feats for cell and bin nodes (one chromosome, multiprocessing)
def generate_feats_one(temp1,temp, total_embed_size, total_chrom_size, c):
	# print (np.sum(temp1 > 0, axis=0))
	mask = np.array(np.sum(temp1 > 0, axis=0) > 10)
	mask = mask.reshape((-1))
	# if type(temp) != np.ndarray:
	# 	temp = np.array(temp.todense())
	size = int(total_embed_size / total_chrom_size * temp.shape[-1]) + 1
		
	temp = temp[:, mask]
	temp /= (np.sum(temp, axis=-1)+1e-15)
	# sparsity = np.sum(temp > 0 ,axis=-1) / temp.shape[-1]
	# print ("sparsity", np.median(sparsity), np.min(sparsity), np.max(sparsity))
	
	U, s, Vt = pca(temp, k=size)  # Automatically centers.
	temp1 =  np.array(U[:, :size] * s[:size])
	
	np.save(os.path.join(temp_dir, "%s_cell_PCA.npy" % c), temp1)
	
def check_sparsity(temp):

	total_reads, total_possible = np.array(np.sum(temp > 0, axis=-1)), temp.shape[1]
	return total_reads, total_possible
	
# generate feats for cell and bin nodes
def generate_feats(smooth_flag=False):
	print ("generating node attributes")
	pool = ProcessPoolExecutor(max_workers=cpu_num)
	chrom2adj = {}
	total_chrom_size = 0.0
	total_linear_chrom_size = 0.0
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
		total_linear_chrom_size += int(math.sqrt(list(temp.shape)[-1]) * res_cell / 1000000)

	if len(chrom_list) > 1:
		total_embed_size = min(max(int(temp.shape[0] * 0.5), int(total_linear_chrom_size * 0.5)), int(temp.shape[0] * 1.3))
	else:
		total_embed_size = int(np.min(temp.shape) * 0.2)
	print ("total_feats_size", total_embed_size)
	p_list = []
	
	# if "batch_id" in config:
	# 	print ("Correcting batch effect 1st round")
	#
		
	for c in chrom_list:
		temp = chrom2adj[c]
		p_list.append(pool.submit(generate_feats_one, temp, temp, total_embed_size, total_chrom_size, c))

	
	
	total_reads, total_possible = 0, 0
	for c in chrom_list:
		if res_cell == 1000000:
			temp = np.load(os.path.join(temp_dir, "%s_cell_%s.npy" % (c, ext_str)), allow_pickle=True).item()
		else:
			temp = np.load(os.path.join(temp_dir, "%s_sparsity_metric_adj.npy" % (c)), allow_pickle=True).item()
		a, b = check_sparsity(temp)
		total_reads += a.reshape((-1))
		total_possible += float(b)
	total_sparsity = total_reads / total_possible
	# print ("sparsity", total_sparsity.shape, total_sparsity, np.median(total_sparsity))
	pool.shutdown(wait=True)

	np.save(os.path.join(temp_dir, "sparsity.npy"), np.array([total_sparsity]))
	
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
		
		cells = np.arange(len(signal_file_one.keys()) - 1)
		for cell in cells:
			one_signal_stack.append(np.array(signal_file_one[str(cell)]))
		one_signal_stack = np.stack(one_signal_stack, axis=0)
		one_signal_stack = StandardScaler().fit_transform(one_signal_stack.reshape((-1, 1))).reshape(
			(len(one_signal_stack), -1))
		one_signal_stack[np.isnan(one_signal_stack)] = 0.0
		
		chrom_list_signal = np.array(signal_file[signal]["bin"]["chrom"])
		for chrom in chrom_list:
			chrom2signals[chrom].append(one_signal_stack[:, chrom_list_signal == chrom])
	
	signal_all = []
	
	for chrom in chrom_list:
		temp = chrom2signals[chrom]
		temp = np.concatenate(temp, axis=-1)
		np.save(os.path.join(temp_dir, "coassay_%s.npy" % chrom), temp)
		signal_all.append(temp)
	signal_all = np.concatenate(signal_all, axis=-1)
	signal_all = PCA(n_components=int(np.min(signal_all.shape) * 0.8)).fit_transform(signal_all)
	np.save(os.path.join(temp_dir, "coassay_all.npy"), signal_all)
	
	
	

	pool = ProcessPoolExecutor(max_workers=int(gpu_num * 1.2))
	for chrom in chrom_list:
		pool.submit(process_signal_one, chrom)
		time.sleep(3)
	pool.shutdown(wait=True)
	

	
	attributes_list = []
	for chrom in chrom_list:
		temp = np.load(os.path.join(temp_dir, "pretrain_coassay_%s.npy" % chrom))
		attributes_list.append(temp)
		
		
		
	attributes_list = np.concatenate(attributes_list, axis=-1)
	attributes_list = StandardScaler().fit_transform(attributes_list)
	

	np.save(os.path.join(temp_dir, "pretrain_coassay.npy"), attributes_list)
	
args = parse_args()

config = get_config(args.config)
chrom_list = config['chrom_list']
res = config['resolution']
res_cell = config['resolution_cell']
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


generate_chrom_start_end()
extract_table()
create_matrix()


optional_smooth_flag = False
generate_feats(optional_smooth_flag)



if "coassay" in config:
	if config["coassay"]:
		process_signal()

if "random_walk" in config:
	if config["random_walk"]:
		impute_all()
