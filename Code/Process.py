import math
import os
import pandas as pd
from Higashi_backend.utils import *
from Higashi_backend.Modules import *
from Higashi_analysis.Higashi_analysis import sqrt_norm, oe
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from scipy.sparse import csr_matrix
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import h5py
import pickle
from sklearn.decomposition import PCA, TruncatedSVD, SparsePCA
from umap import UMAP
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]
cpu_num = multiprocessing.cpu_count()
print ("cpu_num", cpu_num)

import argparse

def parse_args():
	parser = argparse.ArgumentParser(description="Higashi visualization tool")
	parser.add_argument('-c', '--config', type=str, default="./config.JSON")
	return parser.parse_args()


def generate_chrom_start_end():
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

def summarize_distance_vs_count():
	data = np.load(os.path.join(temp_dir, "data.npy"))
	print(data)
	weight = np.load(os.path.join(temp_dir, "weight.npy"))
	cell_num = np.max(data[:, 0]) + 1
	
	record_all = []
	
	for cell in trange(cell_num):
		mask = data[:, 0] == cell
		temp = data[mask]
		temp_weight = weight[mask]
		
		all_ = np.sum(temp_weight)
		larger_25 = np.sum(temp_weight[(temp[:,3] - temp[:, 2]) >= 25])
		larger_50 = np.sum(temp_weight[(temp[:, 3] - temp[:, 2]) >= 50])
		larger_100 = np.sum(temp_weight[(temp[:, 3] - temp[:, 2]) >= 100])
		larger_200 = np.sum(temp_weight[(temp[:, 3] - temp[:, 2]) >= 200])
		record_all.append([all_, larger_25, larger_50, larger_100, larger_200])
	record_all = np.array(record_all)
	data = pd.DataFrame(record_all, columns=['read count', '>=25', '>=50', '>=100', '>=200'])
	print (data)
	data.to_csv("../summary.txt", sep="\t", index=False)
	
	
def extract_table():
	chrom_start_end = np.load(os.path.join(temp_dir, "chrom_start_end.npy"))
	data = pd.read_table(os.path.join(data_dir, "data.txt"), sep="\t")
	# data = data[(data['cell_id'] == 0) & (data['chrom1'] == 'chr1')]
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
	print(data[(data[:, 0] == 0) & (data[:, 1] == 0)].shape, data[(data[:, 0] == 0) & (data[:, 1] == 0)])
	unique, inv, unique_counts = np.unique(data, axis=0, return_inverse=True, return_counts=True)
	new_count = np.zeros_like(unique_counts, dtype='float32')
	for i, iv in enumerate(tqdm(inv)):
		new_count[iv] += count[i]
	print(new_count, unique_counts, data.shape)
	print (unique[(unique[:, 0] == 0) & (unique[:, 1] == 0)].shape, unique[(unique[:, 0] == 0) & (unique[:, 1] == 0)])
	np.save(os.path.join(temp_dir, "data.npy"), unique, allow_pickle=True)
	np.save(os.path.join(temp_dir, "weight.npy"), new_count, allow_pickle=True)


def create_matrix_one_chrom(c, size, cell_size, temp, temp_weight, chrom_start_end, cell_num):
	cell_adj = []
	bin_adj = np.zeros((size, size))
	sparse_list = []
	print (c, temp, temp_weight)
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
		# m = oe(sqrt_norm(m))
		
		
		cell_adj.append(m.reshape((-1)))
	
	cell_adj = np.stack(cell_adj, axis=0)
	cell_adj = csr_matrix(cell_adj)
	print (cell_adj)
	bin_adj /= cell_num
	
	np.save(os.path.join(temp_dir, "%s_sparse_adj.npy" % chrom_list[c]), sparse_list)
	np.save(os.path.join(temp_dir, "%s_cell_adj.npy" % chrom_list[c]), cell_adj)
	np.save(os.path.join(temp_dir, "%s_bin_adj.npy" % chrom_list[c]), bin_adj)


def create_matrix():
	data = np.load(os.path.join(temp_dir, "data.npy"))
	print(data)
	weight = np.load(os.path.join(temp_dir, "weight.npy"))
	cell_num = np.max(data[:, 0]) + 1
	chrom_start_end = np.load(os.path.join(temp_dir, "chrom_start_end.npy"))
	print("chrom_start_end", chrom_start_end)
	
	data_within_chrom_list = []
	weight_within_chrom_list = []
	pool = ProcessPoolExecutor(max_workers=len(chrom_list))
	p_list = []
	for c in range(len(chrom_list)):
		temp = data[data[:, 1] == c]
		temp_weight = weight[data[:, 1] == c]
		
		
		size = chrom_start_end[c, 1] - chrom_start_end[c, 0]
		cell_size = int(math.ceil(size / scale_factor))
		p_list.append(pool.submit(create_matrix_one_chrom, c, size, cell_size, temp, temp_weight, chrom_start_end, cell_num))
		
		# for cell in range(cell_num):
		# 	temp_weight[temp[:, 0] == cell] /= np.sum(temp_weight[temp[:, 0] == cell])
		
		data_within_chrom_list.append(temp)
		weight_within_chrom_list.append(temp_weight)
	pool.shutdown(wait=True)
	bias_feats = []
	for cell in trange(cell_num):
		mask = data[:, 0] == cell
		temp = data[mask]
		temp_weight = weight[mask]
		
		all_ = np.sum(temp_weight)
		bias_feats.append([all_])
	bias_feats = np.log10(np.array(bias_feats))
	bias_feats = MinMaxScaler((0.1, 1)).fit_transform(bias_feats)
	np.save(os.path.join(temp_dir, "bias_feats.npy"), bias_feats)
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
	print(num)
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
	print("data.shape", data.shape, data)
	weight = weight[data[:, 1] != data[:, 2]]
	data = data[data[:, 1] != data[:, 2]]
	
	print("data.shape", data.shape)
	np.save(os.path.join(temp_dir, "filter_data.npy"), data)
	np.save(os.path.join(temp_dir, "filter_weight.npy"), weight)


def run_random_walks(neighbors, alias_1st, nodes, num_walks, walk_len):
	pairs = []
	for node in nodes:
		if len(neighbors[node]) == 0:
			continue
			
		for i in range(num_walks):
			curr_node = node
			for j in range(walk_len):
				if len(neighbors[curr_node]) == 0:
					print ("error, no neighbor", curr_node)
					break
				next_node = neighbors[curr_node][alias_draw(alias_1st[curr_node])]
				# drop self co-occurrences are useless
				if curr_node != node:
					pairs.append((node,curr_node))
				curr_node = next_node
	return np.array(pairs)


def generate_graph(pairs, nodes, weight):
	neighbors = [[] for i in range(np.max(nodes) + 1)]
	alias_1st = [[] for i in range(np.max(nodes) + 1)]
	for i, (pair,w) in enumerate(zip(pairs,weight)):
		for j in range(len(pair)):
			for k in range(j+1, len(pair)):
				neighbors[pair[j]].append(pair[k])
				neighbors[pair[k]].append(pair[j])
				
				alias_1st[pair[j]].append(weight[i])
				alias_1st[pair[k]].append(weight[i])
				
	# for i in range(len(neighbors)):
	# 	if len(neighbors[i]) == 0:
	# 		if i > 0:
	# 			if i-1 not in neighbors:
	# 				neighbors[i].append(i-1)
	# 				alias_1st[i].append(1)
	# 		if i < len(neighbors) - 1:
	# 			if i+1 not in neighbors:
	# 				neighbors[i].append(i+1)
	# 				alias_1st[i].append(1)
		
	for i, a in enumerate(alias_1st):
		alias_1st[i] = alias_setup(a)
		
	return alias_1st, neighbors

	
def process_bulk_mcool():
	chrom_start_end = np.load(os.path.join(temp_dir, "chrom_start_end.npy"))
	print("chrom_start_end", chrom_start_end)
	bed = open(os.path.join(temp_dir, "region.bed"), "w")
	if os.path.exists(bulk_path) and bulk_path != "":
		for root, dirs, files in os.walk(bulk_path):
			for f in files:
				flag = False
				if bulk_hic_identifier in f:
					f = h5py.File(os.path.join(bulk_path, f),"r")
					f = f['resolutions']
					f = f['%d' % res]
					bin_info = f['bins']
					chrom_info = np.array(f['chroms']['name']).astype('str')
					print (chrom_info)
					chrom = np.array(bin_info['chrom'])
					start = np.array(bin_info['start'])
					
					bin2chrom = {}
					bin2realid = {}
					for i in range(len(start)):
						c = chrom_info[chrom[i]]
						s = start[i]
						bin2chrom[i] = c
						bin2realid[i] = math.floor(s / res)
					
					pixel = f['pixels']
					bin1_id = np.array(pixel['bin1_id'])
					bin2_id = np.array(pixel['bin2_id'])
					count = np.array(pixel['count'])
					
					chrom2emptymaps = {}
					
					for i in range(len(chrom_start_end)):
						size = chrom_start_end[i, 1] - chrom_start_end[i, 0]
						chrom2emptymaps[chrom_list[i]] = np.zeros((size ,size))
						
						for j in range(size):
							bed.write("%s\t%d\t%d\n" % (chrom_list[i], j * res, (j + 1) * res))
					
					for i in trange(len(bin1_id)):
						bin1 = bin1_id[i]
						bin2 = bin2_id[i]
						ct = count[i]
						
						chrom1 = str(bin2chrom[bin1])
						chrom2 = str(bin2chrom[bin2])
						
						if chrom1 == chrom2:
							if chrom1 in chrom_list:
								chrom2emptymaps[chrom1][bin2realid[bin1], bin2realid[bin2]] += ct
								chrom2emptymaps[chrom1][bin2realid[bin2], bin2realid[bin1]] += ct
					
					for chrom in chrom_list:
						np.save(os.path.join(temp_dir, "bulk_%s_adj.npy" % chrom), chrom2emptymaps[chrom])
					
					
					
					
def process_bulk():
	chrom_start_end = np.load(os.path.join(temp_dir, "chrom_start_end.npy"))
	bed = open(os.path.join(temp_dir, "region.bed"), "w")
	if os.path.exists(bulk_path) and bulk_path != "":
		for root, dirs, files in os.walk(bulk_path):
			for f in files:
				flag=False
				if bulk_hic_identifier in f:
					for chrom in chrom_list[::-1]:
						if chrom in f.split("_"):
							flag=True
							print ("%s is used for %s" %(f, chrom))
							break
					if flag==False:
						continue
					c = chrom_list.index(chrom)
					tab = pd.read_table(os.path.join(bulk_path, f), header = None)
					tab.columns = ['start', 'end', 'value']
					size = chrom_start_end[c, 1] - chrom_start_end[c, 0]
					
					x = np.array(tab['start'] / res).astype('int')
					y = np.array(tab['end'] / res).astype('int')
					v = np.array(tab['value'])
					
					print(x.shape, x, y, v)
					x = x[~np.isnan(v)]
					y = y[~np.isnan(v)]
					v = v[~np.isnan(v)]
					
					print (x.shape, x, y, v)
					m = np.zeros((size, size), dtype='float32')
					# print(x.shape, y.shape, v.shape, m.shape)
					m[x, y] += v
					m = m + m.T
					np.save(os.path.join(temp_dir, "%s_bulk_adj.npy" % chrom), m)
					
					
					for i in range(size):
						bed.write("%s\t%d\t%d\n" % (chrom, i * res, (i+1)*res))


def signal_one(signal, i, size, aggr_func):
	global binid2nodeid
	s_cell = np.zeros((size))
	if aggr_func == "mean":
		count_cell = np.zeros((size))
	for ind, s in zip(binid2nodeid, signal):
		s_cell[ind] += s
		if aggr_func == "mean":
			count_cell[ind] += 1
	if aggr_func == "mean":
		return s_cell/(count_cell+1e-15), i
	else:
		return s_cell, i


def process_signal():
	signal_file = h5py.File(os.path.join(data_dir, "sc_signal.hdf5"), "r")
	signal_names = config["coassay_signal"]
	get_free_gpu()
	cell_feats = []
	for signal in signal_names:
		one_signal_stack = []
		signal_file_one = signal_file[signal]
		
		cells = np.arange(len(signal_file_one.keys()))
		for cell in cells:
			one_signal_stack.append(np.array(signal_file_one[str(cell)]))
		one_signal_stack = np.stack(one_signal_stack, axis=0).astype('float32')
		one_signal_stack = StandardScaler().fit_transform(one_signal_stack.reshape((-1,1))).reshape((len(one_signal_stack), -1))
		# one_signal_stack = PCA(n_components=1024).fit_transform(one_signal_stack)
		print (one_signal_stack.shape)
		print (one_signal_stack)
		
		
		cell_feats.append(one_signal_stack)
	# label_info = pickle.load(open(os.path.join(data_dir, "label_info.pickle"), "rb"))
	# cell_feats[0] *= np.array(label_info["average_cg_rate"]).reshape((-1, 1))
	# cell_feats[1] *= np.array(label_info["average_ch_rate"]).reshape((-1, 1))
	cell_feats = np.concatenate(cell_feats, axis=1)
	
	# cell_feats1 = np.stack([label_info["average_cg_rate"], label_info["average_ch_rate"]], axis=-1)
	# cell_feats = np.concatenate([cell_feats, cell_feats1], axis=-1)
	
	# cell_feats = StandardScaler().fit_transform(cell_feats.reshape((-1,1))).astype('float32').reshape((len(cell_feats), -1))
	# cell_feats = StandardScaler().fit_transform(cell_feats).astype('float32')
	# pca = PCA(n_components=50)
	# cell_feats_dimred = pca.fit_transform(cell_feats)
	# cell_feats_recon = pca.inverse_transform(cell_feats_dimred)
	# print (np.mean((cell_feats- cell_feats_recon) ** 2))
	
	# autoencoder = TiedAutoEncoder([cell_feats.shape[-1], 4096, 1024], add_activation=True, tied_list=[0, 1], use_bias=True)
	# autoencoder.weight_list[0].data = autoencoder.weight_list[0].data / 10
	# autoencoder.fit(cell_feats, epochs=5000, early_stop=False, sparse=False, batch_size=128)
	# cell_feats = autoencoder.predict(cell_feats)
	
	
	
	# cell_feats  = cell_feats_dimred
	
	np.save(os.path.join(temp_dir, "cell_attributes.npy"), cell_feats)
			
def neighbor_ave_gpu(A, pad):
	if pad == 0:
		return torch.from_numpy(A).float().to(device)
	ll = pad * 2 + 1
	conv_filter = torch.ones(1, 1, ll, ll).to(device)
	B = F.conv2d(torch.from_numpy(A[None, None, :, :]).float().to(device), conv_filter, padding=pad * 2)
	B = B[0, 0, pad:-pad, pad:-pad]
	return (B / float(ll * ll))


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


def impute_gpu(A):
	pad = 1
	rp = 0.5
	# Q = torch.from_numpy(A).float().to(device)
	A = np.log2(A + 1)
	conv_A = neighbor_ave_gpu(A, pad)
	if rp == -1:
		Q2 = conv_A[:]
	else:
		Q2 = random_walk_gpu(conv_A, rp)
	return conv_A.cpu().numpy(), Q2.cpu().numpy()


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

# Below is for generating imputed results
def generate_binpair(shape, min_bin_, max_bin_):
	print(shape)
	samples = []
	for bin1 in range(shape):
		for bin2 in range(bin1 + min_bin_, min(bin1 + max_bin_, shape)):
			samples.append([bin1, bin2])
	samples = np.array(samples)
	return samples

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
	
def impute_all():
	get_free_gpu()
	print("start conv random walk")
	for c in chrom_list:
		a = np.load(os.path.join(temp_dir, "%s_sparse_adj.npy"  % c), allow_pickle=True)
		print("saving")
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
			
		
		samples = generate_binpair(a[0].shape[0], min_bin_, max_bin_)
		with h5py.File(os.path.join(temp_dir, "rw_%s.hdf5" % c), "w") as f:
			f.create_dataset('coordinates', data = samples)
			for i, m in enumerate(tqdm(a)):
				b = np.log2(1+np.array(m.todense()))
				conv_m, m = impute_gpu(np.array(m.todense()))
				v = m[samples[:, 0], samples[:, 1]]
				
				f.create_dataset("cell_%d" % (i), data=v)


def optional_impute_for_cell_adj():
	get_free_gpu()
	print("start conv random walk")
	for c in chrom_list:
		a = np.load(os.path.join(temp_dir, "%s_cell_adj.npy" % c), allow_pickle=True).item()
		a = np.array(a.todense())
		print("saving")
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
		print (a)
		samples = generate_binpair(int(np.sqrt(len(a[0]))), min_bin_, max_bin_)
		impute_list = []
		
		for i, m in enumerate(tqdm(a)):
			m = m.reshape((int(np.sqrt(len(m))), -1))
			b = np.log2(1 + np.array(m))
			
			sparsity = np.sum(b > 0) / (b.shape[0] * b.shape[1])
			while sparsity < 0.5:
				# print ("sparsity", sparsity)
				b = conv_only(b)
				sparsity = np.sum(b > 0) / (b.shape[0] * b.shape[1])
			
			impute_list.append(b.reshape((-1)))
			v = m[samples[:, 0], samples[:, 1]]
		impute_list = np.stack(impute_list, axis=0)
		print(impute_list.shape)
		thres = np.percentile(impute_list, 70, axis=1)
		print(thres)
		impute_list = (impute_list > thres[:, None]).astype('float32') * impute_list
		print(impute_list)
		impute_list = csr_matrix(impute_list)
		sparsity = np.sum(impute_list > 0, axis=-1) / impute_list.shape[-1]
		print("sparsity", sparsity, np.median(sparsity), np.min(sparsity), np.max(sparsity))
		
		np.save(os.path.join(temp_dir, "%s_cell_impute.npy" % c), impute_list)
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


def generate_feats_one(temp1,temp, total_embed_size, total_chrom_size, c):
	print (np.sum(temp1 > 0, axis=0))
	mask = np.array(np.sum(temp1 > 0, axis=0) > 10)
	mask = mask.reshape((-1))
	if type(temp) != np.ndarray:
		temp = np.array(temp.todense())
	size = int(total_embed_size / total_chrom_size * temp.shape[-1]) + 1
	temp = temp[:, mask]
	sparsity = np.sum(temp > 0 ,axis=-1) / temp.shape[-1]
	print ("sparsity", sparsity, np.median(sparsity), np.min(sparsity), np.max(sparsity))
	
	temp = np.log2(temp+1)
	temp /= (np.sum(temp, axis=-1, keepdims=True) + 1e-15)
	temp = quantileNormalize(temp)
	np.save(os.path.join(temp_dir, "%s_cell_quantile.npy" % c), temp)
	#
	
	square_shape = int(math.sqrt(temp.shape[-1]))
	temp1 = TruncatedSVD(n_components=size).fit_transform(temp).astype('float32')
	# temp1 = PCA(n_components=size).fit_transform(temp).astype('float32')
	# temp1 = SparsePCA(n_components=size).fit_transform(temp[:, mask]).astype('float32')
	# temp = UMAP(n_components=size).fit_transform(temp[:, mask]).astype('float32')
	# temp = MinMaxScaler((-1, 1)).fit_transform(temp.reshape((-1, 1))).reshape((len(temp), -1))
	print (temp.shape, temp1.shape)
	# temp1 = temp[:, mask]
	np.save(os.path.join(temp_dir, "%s_cell_PCA.npy" % c), temp1)
	
def generate_feats():
	pool = ProcessPoolExecutor(max_workers=10)
	chrom2adj = {}
	total_chrom_size = 0.0
	for c in chrom_list:
		try:
			temp = np.load(os.path.join(temp_dir, "%s_cell_adj.npy" % c), allow_pickle=True).item()
		except:
			temp = np.load(os.path.join(temp_dir, "%s_cell_adj.npy" % c), allow_pickle=True)
		
		chrom2adj[c] = temp
		total_chrom_size += list(temp.shape)[-1]
	
	if len(chrom_list) > 1:
		total_embed_size = int(temp.shape[0] * 1.3)
	else:
		total_embed_size = int(np.min(temp.shape) * 0.8)
	
	big_feats = []
	p_list = []
	for c in chrom_list:
		temp = chrom2adj[c]
		pool.submit(generate_feats_one, temp, temp, total_embed_size, total_chrom_size, c)
	
	pool.shutdown(wait=True)
	
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
impute_all()
optional_impute_for_cell_adj()
generate_feats()


if "coassay" in config:
	if config["coassay"]:
		process_signal()
# try:
# 	bulk_path = config['bulk_path']
# 	bulk_hic_identifier = config['bulk_hic_identifier']
# 	process_bulk_mcool()
# except:
# 	print ("no bulk hic information")

#