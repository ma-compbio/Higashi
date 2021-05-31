import argparse
from Higashi_backend.Modules import *
from Higashi_analysis.Higashi_analysis import *
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from scipy.sparse import csr_matrix, vstack, SparseEfficiencyWarning, diags, \
	hstack
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import h5py
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from scipy.sparse.csgraph import laplacian
import subprocess
from fbpca import pca
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]


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
	data = data[data['chrom1'] == data['chrom2']].reset_index()
	pos1 = np.array(data['pos1'])
	pos2 = np.array(data['pos2'])
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
	
	# Exclude the chromosomes that showed in the data but are not selected.
	data = np.stack([cell_id, new_chrom, bin1, bin2], axis=-1)
	mask = data[:, 1] >= 0
	count = count[mask]
	data = data[mask]
	
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
			chunksize = int(5e6)
			unique, new_count = [], []
			cell_tab = []
			
			p_list = []
			pool = ProcessPoolExecutor(max_workers=cpu_num)
			print ("First calculating how many lines are there")
			line_count = sum(1 for i in open(os.path.join(data_dir, "data.txt"), 'rb'))
			print("There are %d lines" % line_count)
			bar = trange(line_count, desc=' - Processing ', leave=False, )
			with open(os.path.join(data_dir, "data.txt"), 'r') as csv_file:
				chunk_count = 0
				reader =  pd.read_csv(csv_file, chunksize=chunksize, sep="\t")
				for chunk in reader:
					if len(chunk['cell_id'].unique()) == 1:
						# Only one cell, keep appending
						cell_tab.append(chunk)
					else:
						# More than one cell, append all but the last part
						last_cell = np.array(chunk.tail(1)['cell_id'])[0]
						tails = chunk.iloc[np.array(chunk['cell_id']) != last_cell, :]
						head = chunk.iloc[np.array(chunk['cell_id']) == last_cell, :]
						cell_tab.append(tails)
						cell_tab = pd.concat(cell_tab, axis=0).reset_index()
						p_list.append(pool.submit(data2triplets, cell_tab.copy(deep=True), chrom_start_end, False))
						cell_tab = [head]
						bar.update(n=chunksize)
						bar.refresh()
						
			if len(cell_tab) != 0:
				cell_tab = pd.concat(cell_tab, axis=0).reset_index()
				p_list.append(pool.submit(data2triplets, cell_tab, chrom_start_end, False))
				
			for p in as_completed(p_list):
				u_, n_ = p.result()
				unique.append(u_)
				new_count.append(n_)
				bar.update(n=chunksize)
				bar.refresh()
				
			unique, new_count = np.concatenate(unique, axis=0), np.concatenate(new_count, axis=0)
		else:
			data = pd.read_table(os.path.join(data_dir, "data.txt"), sep="\t")
			# ['cell_name','cell_id', 'chrom1', 'pos1', 'chrom2', 'pos2', 'count']
			unique, new_count = data2triplets(data, chrom_start_end, verbose=True)
	else:
		data = pd.read_table(os.path.join(data_dir, "data.txt"), sep="\t")
		# ['cell_name','cell_id', 'chrom1', 'pos1', 'chrom2', 'pos2', 'count']
		unique, new_count = data2triplets(data, chrom_start_end, verbose=True)
		
	np.save(os.path.join(temp_dir, "data.npy"), unique, allow_pickle=True)
	np.save(os.path.join(temp_dir, "weight.npy"), new_count, allow_pickle=True)


def create_matrix_one_chrom(c, size, cell_size, temp, temp_weight, chrom_start_end, cell_num, pca_flag):
	with warnings.catch_warnings():
		warnings.filterwarnings(
			"ignore", category= SparseEfficiencyWarning
		)
		cell_adj = []
		bin_adj = np.zeros((size, size))
		sparse_list = []
		sparse_list_for_gcn = []
		
		temp_mask = temp
		temp_weight_mask = temp_weight
		for i in range(len(temp_mask)):
			bin_adj[temp_mask[i, 2] - chrom_start_end[c, 0], temp_mask[i, 3] - chrom_start_end[c, 0]] += temp_weight_mask[i]
		bin_adj = bin_adj + bin_adj.T - np.diag(np.diagonal(bin_adj)) + np.eye(len(bin_adj))
		bin_adj = sqrt_norm(bin_adj)
		
		
		if pca_flag:
			size1 = int(0.2 * len(bin_adj))
			U, s, Vt = pca(bin_adj, k=size1)  # Automatically centers.
			bin_adj = np.array(U[:, :size1] * s[:size1])
		read_count = []
		sparsity_metric = []

		total_read_count = np.sum(temp_weight)
		
		for i in trange(cell_num):
			mask = temp[:, 0] == i
			temp2 = (temp[mask, 2:] - chrom_start_end[c, 0])
			temp2_scale = np.floor(temp2 / scale_factor).astype('int')
			temp_weight2 = temp_weight[mask]
			
			read_count.append(np.sum(temp_weight2))
			m1 = csr_matrix((temp_weight2, (temp2[:, 0], temp2[:, 1])), shape=(size, size))
			
			m1 = m1 + m1.T
			sparse_list.append(m1)


			mask1 =  ((temp2[:, 1] - temp2[:, 0]) > 0)

			scale_factor_cell =  total_read_count / cell_num / (np.sum(temp_weight2[mask1]) + 1e-15)
			m2 = csr_matrix((temp_weight2[mask1], (temp2[mask1, 0], temp2[mask1, 1])), shape=(size, size))
			m2 = m2 + m2.T
			m2 = m2 * scale_factor_cell
			m2.data = np.log1p(m2.data)
			
			sparse_list_for_gcn.append(m2)
			
			m = csr_matrix((temp_weight2, (temp2_scale[:, 0], temp2_scale[:, 1])), shape=(cell_size, cell_size))
			m = m + m.T - diags(m.diagonal())
			m = m / (m.sum()+1e-15) * 100000
			diag = m.diagonal(0)
			m1 = m.sum()-diag.sum()
			m = laplacian(m, normed=True)
			m = np.abs(m)
			m = m / (m.sum()+1e-15) * m1
			m += diags(diag)
			
			cell_adj.append(m)
			
			if res_cell != 1000000:
				scale_factor2 = int(1000000 / res)
				size_metric = int(math.ceil(size * res / 1000000))
				temp2_scale = np.floor(temp2 / scale_factor2).astype('int')
				m1 = csr_matrix((temp_weight2, (temp2_scale[:, 0], temp2_scale[:, 1])), shape=(size_metric, size_metric))
				m1 = m1 + m1.T
				sparsity_metric.append(m1.reshape((1, -1)))
				
		cell_adj = np.array(cell_adj)

		if "batch_id" in config :
			batch_id_info = fetch_batch_id(config, "batch_id")
		elif "library_id" in config:
			batch_id_info = fetch_batch_id(config, "library_id")
		else:
			batch_id_info = np.ones((cell_num))
			
		
		bulk = np.sum(cell_adj, axis=0) / len(cell_adj)
		
		bulk_bin = []
		for k in range(bulk.shape[0]):
			bulk_bin.append(np.sum(bulk[k, :]) / (bulk.shape[0]))
		bulk_bin = np.array(bulk_bin)
		
		batches = np.unique(batch_id_info)
		for index, b in enumerate(batches):
			b_bin = []
			b_c = np.sum(cell_adj[batch_id_info == b], axis=0) / np.sum(batch_id_info == b)
			for k in range(b_c.shape[0]):
				b_bin.append(np.sum(b_c[k, :] ) / b_c.shape[0])
			b_bin = np.array(b_bin)
			
			
			if  spearmanr(b_bin[bulk_bin > 0.0], bulk_bin[bulk_bin > 0.0])[0] < 0.8:
				print (c, "correct be for batch", b, spearmanr(b_bin[bulk_bin > 0.0], bulk_bin[bulk_bin > 0.0]))
				for i in np.where(batch_id_info == b)[0]:
					m = cell_adj[i]
					row_sums = b_bin + 1e-15
					row_indices, col_indices = m.nonzero()
					m.data /= row_sums[row_indices]
					m.data *= bulk_bin[row_indices]
					
					cell_adj[i] = m.reshape((1, -1))
			else:
				for i in np.where(batch_id_info == b)[0]:
					m = cell_adj[i]
					cell_adj[i] = m.reshape((1, -1))

		cell_adj = vstack(cell_adj).tocsr()
		
		if res_cell != 1000000:
			sparsity_metric = vstack(sparsity_metric).tocsr()
			a, b = check_sparsity(sparsity_metric)
		else:
			a, b = check_sparsity(cell_adj)

		np.save(os.path.join(raw_dir, "%s_sparse_adj.npy" % chrom_list[c]), sparse_list)
		
		return np.array(read_count).reshape((-1)), np.array(sparse_list_for_gcn), c, a, b, bin_adj, cell_adj


def create_or_overwrite(file, name="", data=0):
	if name in file.keys():
		file[name][...] = data
	else:
		file.create_dataset(name=name, data=data)

# Generate matrices for feats and baseline
def create_matrix():
	print ("generating contact maps for baseline")
	data = np.load(os.path.join(temp_dir, "data.npy"))
	weight = np.load(os.path.join(temp_dir, "weight.npy"))



	cell_num = np.max(data[:, 0]) + 1
	chrom_start_end = np.load(os.path.join(temp_dir, "chrom_start_end.npy"))

	data_within_chrom_list = []
	weight_within_chrom_list = []
	pool = ProcessPoolExecutor(max_workers=23)
	p_list = []

	cell_feats = [[] for i in range(len(chrom_list))]
	sparse_chrom_list = [[] for i in range(len(chrom_list))]

	save_mem = False

	print (len(data), save_mem)
	pca_flag = False

	total_reads, total_possible = 0, 0
	with h5py.File(os.path.join(temp_dir, "node_feats.hdf5"), "w") as save_file:
		for c in range(len(chrom_list)):
			temp = data[data[:, 1] == c]
			temp_weight = weight[data[:, 1] == c]

			if len(temp) > 7e7:
				save_mem = True
			else:
				save_mem = False
			
			print (chrom_list[c], "save_mem", save_mem)
			size = chrom_start_end[c, 1] - chrom_start_end[c, 0]

			if size >= 1000:
				pca_flag = True

			cell_size = int(math.ceil(size / scale_factor))

			data_within_chrom_list.append(np.copy(temp))
			weight_within_chrom_list.append(np.copy(temp_weight))



			chrom2celladj = {}
			
			total_linear_chrom_size = 0.0

			if save_mem:
				chrom_count, non_diag_sparse, c, a, b, bin_adj, cell_adj = create_matrix_one_chrom( c, size, cell_size, temp, temp_weight, chrom_start_end, cell_num, pca_flag)
				if "%d" % c in save_file.keys():
					save_file["%d" % c][...] = bin_adj
				else:
					save_file.create_dataset(name="%d" % c, data=bin_adj)



				total_reads += a.reshape((-1))
				total_possible += float(b)
				cell_feats[c] = chrom_count
				sparse_chrom_list[c] = non_diag_sparse

				chrom2celladj[c] = cell_adj
				total_linear_chrom_size += int(math.sqrt(list(cell_adj.shape)[-1]) * res_cell / 1000000)

			else:
				p_list.append(
					pool.submit(create_matrix_one_chrom, c, size, cell_size, temp, temp_weight, chrom_start_end, cell_num, pca_flag))


		if len(p_list) > 0:
			for p in as_completed(p_list):
				chrom_count, non_diag_sparse, c, a, b, bin_adj, cell_adj = p.result()
				if "%d" % c in save_file.keys():
					save_file["%d" % c][...] = bin_adj
				else:
					save_file.create_dataset(name="%d" % c, data=bin_adj)

				chrom2celladj[c] = cell_adj
				total_linear_chrom_size += int(math.sqrt(list(cell_adj.shape)[-1]) * res_cell / 1000000)


				total_reads += a.reshape((-1))
				total_possible += float(b)
				cell_feats[c] = chrom_count
				sparse_chrom_list[c] = non_diag_sparse
		pool.shutdown(wait=True)

		pool = ProcessPoolExecutor(max_workers=cpu_num)
		if len(chrom_list) > 1:

			total_embed_size = min(max(int(cell_adj.shape[0] * 0.5), int(total_linear_chrom_size * 0.5)),
								   int(cell_adj.shape[0] * 1.3))
		else:
			total_embed_size = int(np.min(cell_adj.shape) * 0.8)
		print("total_feats_size", total_embed_size)
		p_list = []

		for c in range(len(chrom_list)):
			temp = chrom2celladj[c]
			p_list.append(pool.submit(generate_feats_one, temp, total_embed_size, total_linear_chrom_size, c))

		if "cell" not in save_file.keys():
			save_file_cell = save_file.create_group("cell")
		else:
			save_file_cell = save_file["cell"]

		for p in as_completed(p_list):
			temp1, c = p.result()

			if "%d" % c in save_file_cell.keys():
				save_file_cell["%d" % c][...]= temp1
			else:
				save_file_cell.create_dataset(name="%d" % c, data=temp1)

		pool.shutdown(wait=True)

		total_sparsity = total_reads / total_possible
		print("sparsity", total_sparsity.shape, total_sparsity, np.median(total_sparsity))

		create_or_overwrite(save_file, "sparsity", data=total_sparsity)

		sparse_chrom_list = np.array(sparse_chrom_list)
		np.save(os.path.join(temp_dir, "sparse_nondiag_adj_nbr_1.npy"), sparse_chrom_list)
		cell_feats = np.stack(cell_feats, axis=-1)
		pool.shutdown(wait=True)

		create_or_overwrite(save_file, "extra_cell_feats", data=cell_feats)

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

		# np.save(os.path.join(temp_dir, "num.npy"), num)
		create_or_overwrite(save_file, "num", data=num)

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

		create_or_overwrite(save_file, "start_end_dict", data=start_end_dict)
		create_or_overwrite(save_file, "id2chrom", data=id2chrom)

		mask = data[:, 1] != data[:, 2]
		weight = weight[mask]
		data = data[mask]
		chrom_info = chrom_info[mask]

		# Add 1 for padding idx
		create_or_overwrite(save_file, "data", data=data + 1)
		create_or_overwrite(save_file, "chrom", data=chrom_info)
		create_or_overwrite(save_file, "weight", data=weight)


		distance = data[:, 2] - data[:, 1]
		info = pd.DataFrame({'dis': distance, 'weight': weight, 'cell':data[:, 0]})
		info1 = info.groupby(by='dis').mean().reset_index()
		print(info1)
		max_bin1 = int(np.max(num[2:]))
		distance2weight = np.zeros((max_bin1, 1), dtype='float32')
		distance2weight[np.array(info1['dis']), 0] = np.array(info1['weight'])
		create_or_overwrite(save_file, "distance2weight", data=distance2weight)

		info1 = info.groupby(by='cell').mean().reset_index()
		print(info1)
		cell2weight = np.zeros((num[1], 1), dtype='float32')
		cell2weight[np.array(info1['cell']), 0] = np.array(info1['weight'])
		create_or_overwrite(save_file, "cell2weight", data=cell2weight)





# Code from scHiCluster
def neighbor_ave_gpu(A, pad, device):
	if pad == 0:
		return torch.from_numpy(A).float().to(device)
	ll = pad * 2 + 1
	conv_filter = torch.ones(1, 1, ll, ll).to(device)
	B = F.conv2d(torch.from_numpy(A[None, None, :, :]).float().to(device), conv_filter, padding=pad * 2)
	B = B[0, 0, pad:-pad, pad:-pad]
	return (B / float(ll * ll))

# Code from scHiCluster
def random_walk_gpu(A, rp, epochs=60, device=None):
	ngene, _ = A.shape
	A = A.float()
	A = A - torch.diag(torch.diag(A))
	A = A + torch.diag((torch.sum(A, 0) == 0).float())
	
	P = torch.div(A, torch.sum(A, 0) + 1e-15)
	Q = torch.eye(ngene).to(device)
	I = torch.eye(ngene).to(device)
	for i in range(epochs):
		# print ("actually running")
		Q_new = (1 - rp) * I + rp * torch.mm(Q, P)
		delta = torch.norm(Q - Q_new, 2)
		Q = Q_new
		if delta < 1e-6:
			break
	Q = Q - torch.diag(torch.diag(Q))
	return Q

# Code from scHiCluster
def impute_gpu(A, device=None):
	pad = 1
	rp = 0.5
	# pad = 0
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	conv_A = neighbor_ave_gpu(A, pad, device=device)
	if rp == -1:
		Q2 = conv_A[:]
	else:
		Q2 = random_walk_gpu(conv_A, rp, device=device)
	return Q2.cpu().numpy()

# Code from scHiCluster
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
	sc_list = []
	for c in impute_list:
		a = np.load(os.path.join(raw_dir, "%s_sparse_adj.npy"  % c), allow_pickle=True)
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
		with h5py.File(os.path.join(rw_dir, "rw_%s.hdf5" % c), "w") as f:
			f.create_dataset('coordinates', data = samples)
			for i, m in enumerate(tqdm(a)):
				m = np.array(m.todense())
				m = impute_gpu(np.array(m))
				v = m[samples[:, 0], samples[:, 1]]
				#
				f.create_dataset("cell_%d" % (i), data=v)
				


# generate feats for cell and bin nodes (one chromosome, multiprocessing)
def generate_feats_one(temp, total_embed_size, total_chrom_size, c):
	if temp.shape[0] > 3:
		mask = np.array(np.sum(temp > 0, axis=0) > min(5, temp.shape[0]-2))
		mask = mask.reshape((-1))
		
		length = int(np.sqrt(temp.shape[-1]) / 1000000 * res_cell)
		size = int(total_embed_size / total_chrom_size * length) + 1
		
		temp = normalize(temp, norm='l1', axis=1) * 100000
		# print (temp.shape, total_embed_size, total_chrom_size, length, size)
		temp = temp[:, mask]
		
		
		U, s, Vt = pca(temp, k=size)  # Automatically centers.
		temp1 =  np.array(U[:, :size] * s[:size])
	else:
		temp1 = np.eye(temp1.shape[0])

	return temp1, c
	# np.save(os.path.join(temp_dir, "%s_cell_PCA.npy" % c), temp1)
	
def check_sparsity(temp):

	total_reads, total_possible = np.array(np.sum(temp > 0, axis=-1)), temp.shape[1]
	return total_reads, total_possible
	
	
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

def scool_rwr():
	import cooler
	from Higashi2Scool import HigashiDict, skip_start_end
	chrom2info = {}
	
	
	off_set = 0
	
	bins_chrom = []
	bins_start = []
	bins_end = []
	
	cell_list = []
	for chrom_index, chrom in enumerate(chrom_list):

		impute_f = h5py.File(os.path.join(rw_dir, "rw_%s.hdf5" % (chrom)), "r")
		
		origin_sparse = np.load(os.path.join(raw_dir, "%s_sparse_adj.npy" % chrom), allow_pickle=True)
		size = origin_sparse[0].shape[0]
		mask_start, mask_end = skip_start_end(config, chrom)
		del origin_sparse
		
		coordinates = np.array(impute_f['coordinates']).astype('int')
		xs, ys = coordinates[:, 0], coordinates[:, 1]
		m1 = np.zeros((size, size))
		chrom2info[chrom] = [size, mask_start, mask_end, impute_f, xs, ys, m1, off_set]
		off_set += size
		bins_chrom += [chrom] * size
		bins_start.append(np.arange(size) * res)
		bins_end.append(np.arange(size) * res + res)
		
		if chrom_index == 0:
			for i in range(len(list(impute_f.keys())) - 1):
				cell_list.append("cell_%d" % i)
	
	bins = pd.DataFrame({'chrom': bins_chrom, 'start': np.concatenate(bins_start), 'end': np.concatenate(bins_end)})
	cell_name_pixels_dict = HigashiDict(chrom2info, cell_list, chrom_list)
	
	
	print("Start creating scool")
	
	
	cooler.create_scool(os.path.join(rw_dir, "rw_impute.scool"), bins,
	                    cell_name_pixels_dict, dtypes={'count': 'float32'}, ordered=True)


def scool_raw():
	import cooler
	
	off_set = 0
	
	bins_chrom = []
	bins_start = []
	bins_end = []
	
	cell_list = []
	cell_name_pixels_dict = {}
	for chrom_index, chrom in enumerate(chrom_list):
		
		origin_sparse = np.load(os.path.join(raw_dir, "%s_sparse_adj.npy" % chrom), allow_pickle=True)
		size = origin_sparse[0].shape[0]
		
		
		bins_chrom += [chrom] * size
		bins_start.append(np.arange(size) * res)
		bins_end.append(np.arange(size) * res + res)
		
		
		if chrom_index == 0:
			for i in range(len(origin_sparse)):
				cell_list.append("cell_%d" % i)
			
			
		
		for i in range(len(origin_sparse)):
			xs, ys = origin_sparse[i].nonzero()
			
			v = np.array(origin_sparse[i].data).reshape((-1))
			
			mask = ys >= xs
			temp = pd.DataFrame(
					{'bin1_id': xs[mask] + off_set, 'bin2_id': ys[mask] + off_set, 'count':v[mask]})
			if 'cell_%d' % i not in cell_name_pixels_dict:
				cell_name_pixels_dict['cell_%d' % i] = temp
			else:
				cell_name_pixels_dict['cell_%d' % i] = pd.concat([cell_name_pixels_dict['cell_%d' % i], temp], axis=0)
		off_set += size
			
			
	print("Start creating scool")
	
	bins = pd.DataFrame(
		{'chrom': bins_chrom, 'start': np.concatenate(bins_start), 'end': np.concatenate(bins_end)})
	cooler.create_scool(os.path.join(temp_dir, "raw.scool"), bins, cell_name_pixels_dict,
	                    dtypes={'count': 'float32'}, ordered=True)
		
		
		
args = parse_args()

config = get_config(args.config)
chrom_list = config['chrom_list']
impute_list = config['impute_list']
res = config['resolution']
res_cell = config['resolution_cell']
scale_factor = int(res_cell / res)
print ("scale_factor", scale_factor)


data_dir = config['data_dir']
temp_dir = config['temp_dir']
if not os.path.exists(temp_dir):
	os.mkdir(temp_dir)

raw_dir = os.path.join(temp_dir, "raw")
if not os.path.exists(raw_dir):
	os.mkdir(raw_dir)


rw_dir = os.path.join(temp_dir, "rw")
if not os.path.exists(rw_dir):
	os.mkdir(rw_dir)

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


if "coassay" in config:
	if config["coassay"]:
		process_signal()

if "random_walk" in config:
	if config["random_walk"]:
		impute_all()


# scool_rwr()
# scool_raw()