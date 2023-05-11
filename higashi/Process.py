import argparse
import shutil

try:
	from Higashi_backend.Modules import *
	from Higashi_analysis.Higashi_analysis import *
except:
	try:
		from .Higashi_backend.Modules import *
		from .Higashi_analysis.Higashi_analysis import *
	except:
		raise EOFError

import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from scipy.sparse import csr_matrix, vstack, SparseEfficiencyWarning, diags, \
	hstack
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import subprocess
from scipy.ndimage import gaussian_filter

try:
	get_ipython()
	from tqdm.notebook import tqdm, trange
except:
	pass


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
	

def create_dir(config):
	temp_dir = config['temp_dir']
	if not os.path.exists(temp_dir):
		os.mkdir(temp_dir)
	
	raw_dir = os.path.join(temp_dir, "raw")
	if not os.path.exists(raw_dir):
		os.mkdir(raw_dir)
	
	
	rw_dir = os.path.join(temp_dir, "rw")
	if not os.path.exists(rw_dir):
		os.mkdir(rw_dir)

	embed_dir = os.path.join(temp_dir, "embed")
	if not os.path.exists(embed_dir):
		os.mkdir(embed_dir)
	
# Generate a indexing table of start and end id of each chromosome
def generate_chrom_start_end(config):
	# fetch info from config
	genome_reference_path = config['genome_reference_path']
	chrom_list = config['chrom_list']
	res = config['resolution']
	temp_dir = config['temp_dir']
	
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
	
	# print("chrom_start_end", chrom_start_end)
	np.save(os.path.join(temp_dir, "chrom_start_end.npy"), chrom_start_end)
	
	
def data2triplets(config, data, chrom_start_end, verbose):
	# fetch info from config
	res = config['resolution']
	chrom_list = config['chrom_list']
	if 'downsample' in config:
		downsample = config['downsample']
	else:
		downsample = 1.0
	
	if type(data) is tuple:
		file, cell_id = data
		if "header_included" in config:
			if config['header_included']:
				tab = pd.read_table(file, sep="\t")
			else:
				tab = pd.read_table(file, sep="\t", header=None)
				tab.columns = config['contact_header']
		else:
			tab = pd.read_table(file, sep="\t", header=None)
			tab.columns = config['contact_header']
		tab['cell_id'] = cell_id
		if 'count' not in tab.columns:
			tab['count'] = 1
		data = tab
		data = data[(((np.abs(data['pos2'] - data['pos1']) >= 2500) | (np.abs(data['pos2'] - data['pos1']) == 0)) & (data['chrom1'] == data['chrom2'])) | (
						data['chrom1'] != data['chrom2'])].reset_index()
		
		
	pos1 = np.array(data['pos1'])
	pos2 = np.array(data['pos2'])
	bin1 = np.floor(pos1 / res).astype('int')
	bin2 = np.floor(pos2 / res).astype('int')
	
	chrom1, chrom2 = np.array(data['chrom1'].values), np.array(data['chrom2'].values)
	cell_id = np.array(data['cell_id'].values).astype('int')
	count = np.array(data['count'].values)
	
	del data
	
	new_chrom1 = np.ones_like(bin1, dtype='int') * -1
	new_chrom2 = np.ones_like(bin1, dtype='int') * -1
	for i, chrom in enumerate(chrom_list):
		mask = (chrom1 == chrom)
		new_chrom1[mask] = i
		# Make the bin id for chromosome 2 starts at the end of the chromosome 1, etc...
		bin1[mask] += chrom_start_end[i, 0]
		mask = (chrom2 == chrom)
		new_chrom2[mask] = i
		bin2[mask] += chrom_start_end[i, 0]
	
	# Exclude the chromosomes that showed in the data but are not selected.
	data = np.stack([cell_id, new_chrom1, new_chrom2, bin1, bin2], axis=-1)
	mask = (data[:, 1] >= 0) & (data[:, 2] >= 0)
	count = count[mask]
	data = data[mask]
	
	if downsample < 1:
		# print ("downsample at", downsample)
		index = np.random.permutation(len(data))[:int(downsample * len(data))]
		count = count[index]
		data = data[index]
	
	unique, inv, unique_counts = np.unique(data, axis=0, return_inverse=True, return_counts=True)
	new_count = np.zeros_like(unique_counts, dtype='float32')
	func1 = tqdm if verbose else pass_
	for i, iv in enumerate(func1(inv)):
		new_count[iv] += count[i]
	
		
	return unique, new_count

def split_intra_inter(u_, n_):
	intra_ = u_[:, 1] == u_[:, 2]
	inter_ = u_[:, 1] != u_[:, 2]
	
	intra_data = u_[intra_]
	intra_count = n_[intra_]
	intra_data = intra_data[:, [0, 1, 3, 4]]
	bin1, bin2 = intra_data[:, 2], intra_data[:, 3]
	new_bin1 = np.minimum(bin1, bin2)
	new_bin2 = np.maximum(bin1, bin2)
	intra_data[:, 2] = new_bin1
	intra_data[:, 3] = new_bin2
	
	inter_data = u_[inter_]
	inter_count = n_[inter_]
	return intra_data, intra_count, inter_data, inter_count

# Extra the data.txt table
# Memory consumption re-optimize
def extract_table(config):
	if 'keep_inter' in config:
		keep_inter = config['keep_inter']
	else:
		keep_inter = False
		
	# fetch info from config
	data_dir = config['data_dir']
	temp_dir = config['temp_dir']
	if 'input_format' in config:
		input_format = config['input_format']
	else:
		input_format = 'higashi_v1'
	
	intra_data_all = []
	intra_count_all = []
	inter_data_all = []
	inter_count_all = []
	
	chrom_start_end = np.load(os.path.join(temp_dir, "chrom_start_end.npy"))
	if input_format == 'higashi_v1':
		print ("extracting from data.txt")
		if "structured" in config:
			if config["structured"]:
				chunksize = int(5e6)
				# unique, new_count = [], []
				cell_tab = []
				
				p_list = []
				pool = ProcessPoolExecutor(max_workers=cpu_num)
				print ("First calculating how many lines are there")
				line_count = sum(1 for i in open(os.path.join(data_dir, "data.txt"), 'rb'))
				print("There are %d lines" % line_count)
				bar = trange(line_count, desc=' - Processing ', leave=False, )
				with open(os.path.join(data_dir, "data.txt"), 'r') as csv_file:
					chunk_count = 0
					reader = pd.read_csv(csv_file, chunksize=chunksize, sep="\t")
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
							p_list.append(pool.submit(data2triplets, config, cell_tab.copy(deep=True), chrom_start_end, False))
							cell_tab = [head]
							bar.update(n=chunksize)
							bar.refresh()
							
				if len(cell_tab) != 0:
					cell_tab = pd.concat(cell_tab, axis=0).reset_index()
					p_list.append(pool.submit(data2triplets, config, cell_tab, chrom_start_end, False))
					
				for p in as_completed(p_list):
					u_, n_ = p.result()
					intra_data, intra_count, inter_data, inter_count = split_intra_inter(u_, n_)
					intra_data_all.append(intra_data)
					intra_count_all.append(intra_count)
					if keep_inter:
						inter_data_all.append(inter_data)
						inter_count_all.append(inter_count)
					bar.update(n=chunksize)
					bar.refresh()
					
				intra_data = np.concatenate(intra_data_all, axis=0)
				intra_count = np.concatenate(intra_count_all, axis=0)
				if keep_inter:
					inter_data = np.concatenate(inter_data_all, axis=0)
					inter_count = np.concatenate(inter_count_all, axis=0)
				
			else:
				data = pd.read_table(os.path.join(data_dir, "data.txt"), sep="\t")
				# ['cell_name','cell_id', 'chrom1', 'pos1', 'chrom2', 'pos2', 'count']
				unique, new_count = data2triplets(config, data, chrom_start_end, verbose=True)
				intra_data, intra_count, inter_data, inter_count = split_intra_inter(unique, new_count)

		else:
			data = pd.read_table(os.path.join(data_dir, "data.txt"), sep="\t")
			# ['cell_name','cell_id', 'chrom1', 'pos1', 'chrom2', 'pos2', 'count']
			unique, new_count = data2triplets(config, data, chrom_start_end, verbose=True)
			intra_data, intra_count, inter_data, inter_count = split_intra_inter(unique, new_count)
			
			
	elif input_format == 'higashi_v2':
		print ("extracting from filelist.txt")
		with open(os.path.join(data_dir, "filelist.txt"), "r") as f:
			lines = f.readlines()
			filelist = [line.strip() for line in lines]
		bar = trange(len(filelist))
		
		p_list = []
		pool = ProcessPoolExecutor(max_workers=cpu_num)
		
		
		for cell_id, file in enumerate(filelist):
			p_list.append(pool.submit(data2triplets, config, (file, cell_id), chrom_start_end, False))
			
		for p in as_completed(p_list):
			u_, n_ = p.result()
			intra_data, intra_count, inter_data, inter_count = split_intra_inter(u_, n_)
			intra_data_all.append(intra_data)
			intra_count_all.append(intra_count)
			if keep_inter:
				inter_data_all.append(inter_data)
				inter_count_all.append(inter_count)
			bar.update(n=1)
			bar.refresh()
			
			# tab = tab[((np.abs(tab['pos2'] - tab['pos1']) >= 2500) & (tab['chrom1'] == tab['chrom2'])) | (
			# 			tab['chrom1'] != tab['chrom2'])].reset_index()
			# u_, n_= data2triplets(config, tab, chrom_start_end, verbose=False)
			# intra_data, intra_count, inter_data, inter_count = split_intra_inter(u_, n_)
			# intra_data_all.append(intra_data)
			# inter_data_all.append(inter_data)
			# intra_count_all.append(intra_count)
			# inter_count_all.append(inter_count)
			# bar.update(1)
		intra_data = np.concatenate(intra_data_all, axis=0)
		intra_count = np.concatenate(intra_count_all, axis=0)
		if keep_inter:
			inter_data = np.concatenate(inter_data_all, axis=0)
			inter_count = np.concatenate(inter_count_all, axis=0)
		bar.close()
		
				
		
	else:
		print ("invalid input format")
		raise EOFError
	
	
	
	
	np.save(os.path.join(temp_dir, "data.npy"), intra_data, allow_pickle=True)
	np.save(os.path.join(temp_dir, "weight.npy"), intra_count.astype('float32'), allow_pickle=True)
	if keep_inter:
		np.save(os.path.join(temp_dir, "inter_data.npy"), inter_data, allow_pickle=True)
		np.save(os.path.join(temp_dir, "inter_weight.npy"), inter_count.astype('float32'), allow_pickle=True)


def create_matrix_one_chrom(config, c, size, cell_size, temp, temp_weight, chrom_start_end, cell_num, total_part_num=1, part_num=0, per_cell_read=1000):
	# fetch infor from config
	res = config['resolution']
	res_cell = config['resolution_cell']
	scale_factor = int(res_cell / res)
	chrom_list = config['chrom_list']
	temp_dir = config['temp_dir']
	raw_dir = os.path.join(temp_dir, "raw")
	
	
	with warnings.catch_warnings():
		warnings.filterwarnings(
			"ignore", category= SparseEfficiencyWarning
		)
		cell_adj = []
		qc_list = []
		sparse_list = []
		sparse_list_for_gcn = []
		
		read_count = []
		
		a = []
		
		if type(cell_num) is int:
			bar = range(cell_num)
			cell_id = np.arange(cell_num)
		else:
			bar = cell_num
			cell_id = cell_num
			
		for i in bar:
			mask = temp[:, 0] == i
			temp2 = (temp[mask, 2:] - chrom_start_end[c, 0])
			temp2_scale = np.floor(temp2 / scale_factor).astype('int')
			temp_weight2 = temp_weight[mask]
			
			read_count.append(np.sum(temp_weight2))
			m1 = csr_matrix((temp_weight2, (temp2[:, 0], temp2[:, 1])), shape=(size, size), dtype='float32')
			qc_list.append(np.sum(temp2[:, 0] != temp2[:, 1]))
			m1 = m1 + m1.T
			sparse_list.append(m1)


			mask1 =  ((temp2[:, 1] - temp2[:, 0]) > 0)

			read_count_norm =  per_cell_read / (np.sum(temp_weight2[mask1]) + 1e-15)
			m2 = csr_matrix((temp_weight2[mask1], (temp2[mask1, 0], temp2[mask1, 1])), shape=(size, size), dtype='float32')
			m2 = m2 + m2.T
			m2 += diags(np.array(m2.sum(axis=-1) == 0).reshape((-1)).astype('float32'))
			m2 = m2 * read_count_norm
			
			sparse_list_for_gcn.append(m2)
			
			# no more laplacian processing
			m = csr_matrix((temp_weight2, (temp2_scale[:, 0], temp2_scale[:, 1])), shape=(cell_size, cell_size), dtype='float32')
			m = m + m.T
			m = csr_matrix(gaussian_filter((m).astype(np.float32).toarray(), 1, order=0, truncate=1))
			m = m / (m.sum() + 1e-15)
			# diag = m.diagonal(0)
			# m1 = m.sum() - diag.sum()
			# m = laplacian(m, normed=True)
			# m = np.abs(m)
			# m = m / (m.sum() + 1e-15) * m1
			# m += diags(diag)
			
			cell_adj.append(m)
			
			if res_cell != 1000000:
				scale_factor2 = int(1000000 / res)
				size_metric = int(math.ceil(size * res / 1000000))
				temp2_scale = np.floor(temp2 / scale_factor2).astype('int')
			else:
				size_metric = int(math.ceil(size * res / 1000000))
			a.append(len(np.unique(temp2_scale, axis=0)))
			b = int(size_metric * size_metric / 2)
			
		cell_adj = np.array(cell_adj)
		
		if total_part_num == 1:
			np.save(os.path.join(raw_dir, "%s_sparse_adj.npy" % chrom_list[c]), sparse_list)
			np.save(os.path.join(temp_dir, "temp", "sparse_gcn_%s.npy" % chrom_list[c]), np.array(sparse_list_for_gcn), allow_pickle=True)
			np.save(os.path.join(temp_dir, "temp", "cell_adj_%s.npy" % chrom_list[c]), cell_adj,
			        allow_pickle=True)
		else:
			np.save(os.path.join(temp_dir, "temp", "%s_sparse_adj_part_%d.npy" % (chrom_list[c], part_num)), sparse_list)
			np.save(os.path.join(temp_dir, "temp", "sparse_gcn_%s_part_%d.npy" % (chrom_list[c], part_num)), np.array(sparse_list_for_gcn),
			        allow_pickle=True)
			np.save(os.path.join(temp_dir, "temp", "cell_adj_%s_part_%d.npy" % (chrom_list[c], part_num)), cell_adj,
			        allow_pickle=True)
		return np.array(read_count).reshape((-1)), c, np.array(a), b, total_part_num, part_num, cell_id, np.asarray(qc_list)


def create_or_overwrite(file, name="", data=0):
	if name in file.keys():
		if type(data) is np.ndarray:
			shape1 =  file[name].shape
			flag = True
			for i in range(len(shape1)):
				if shape1[i] != data.shape[i]:
					flag = False
			if flag:
				file[name][...] = data
			else:
				del file[name]
				file.create_dataset(name=name, data=data)
		else:
			del file[name]
			file.create_dataset(name=name, data=data)
	else:
		file.create_dataset(name=name, data=data)


def create_inter_matrix(config, cell_num):
	# fetch info from config
	chrom_list = config['chrom_list']
	temp_dir = config['temp_dir']
	raw_dir = os.path.join(temp_dir, "raw")
	
	print ("generating interchromosomal contacts")
	data = np.load(os.path.join(temp_dir, "inter_data.npy"))
	weight = np.load(os.path.join(temp_dir, "inter_weight.npy"))
	chrom_start_end = np.load(os.path.join(temp_dir, "chrom_start_end.npy"))
	
	n_bins = chrom_start_end[-1][-1]
	import sparse
	coor = data[:, [0, 3, 4]].T
	print (coor, n_bins, np.min(coor, axis=1), np.max(coor, axis=1))
	new_matrix = sparse.COO(coor, weight, shape=(cell_num, n_bins, n_bins))
	print (new_matrix.shape)
	
	dense_in_cell = []
	
	for cell in trange(cell_num):
		cell_matrix = new_matrix[cell]
		cell_matrix = cell_matrix + cell_matrix.T
		cell_matrix = cell_matrix.tocsr()
		dense_in_cell.append(cell_matrix)
	del new_matrix
	
	for c in trange(len(chrom_list)):
		start, end = chrom_start_end[c, 0], chrom_start_end[c, 1]
		chrom_cell_list = np.array([mtx[start:end, :] for mtx in dense_in_cell], dtype='object')
		np.save(os.path.join(raw_dir, "%s_sparse_inter_adj.npy" % chrom_list[c]), chrom_cell_list)

# Generate matrices for feats and baseline
def create_matrix(config, disable_mpl=False):
	# fetch info from config
	chrom_list = config['chrom_list']
	temp_dir = config['temp_dir']
	raw_dir = os.path.join(temp_dir, "raw")
	res = config['resolution']
	res_cell = config['resolution_cell']
	scale_factor = int(res_cell / res)
	if 'keep_inter' in config:
		keep_inter = config['keep_inter']
	else:
		keep_inter = False
	
	print("generating contact maps for baseline")
	data = np.load(os.path.join(temp_dir, "data.npy"))
	weight = np.load(os.path.join(temp_dir, "weight.npy"))
	
	print ("data loaded")
	if not os.path.exists(os.path.join(temp_dir, "temp")):
		os.mkdir(os.path.join(temp_dir, "temp"))

	cell_num = int(np.max(data[:, 0]) + 1)
	
	if keep_inter:
		create_inter_matrix(config, cell_num=cell_num)
	
	chrom_start_end = np.load(os.path.join(temp_dir, "chrom_start_end.npy"))
	
	data_within_chrom_list = []
	weight_within_chrom_list = []
	pool = ProcessPoolExecutor(max_workers=cpu_num)
	p_list = []
	
	cell_feats = [[] for i in range(len(chrom_list))]
	sparse_chrom_list = [[] for i in range(len(chrom_list))]
	qc_list = [[] for i in range(len(chrom_list))]
	
	save_mem = False
	
	print(len(data), save_mem)
	# pca_flag = False

	total_reads, total_possible = np.zeros(cell_num), 0
	with h5py.File(os.path.join(temp_dir, "node_feats.hdf5"), "a") as save_file:
		c2total_part_num = {}
		binadj_dict = {}
		for c in range(len(chrom_list)):
			mask = data[:, 1] == c
			temp = data[mask]
			temp_weight = weight[mask]

			data = data[~mask]
			weight = weight[~mask]

			if len(temp) > 3e6:
				save_mem = True
			else:
				save_mem = False

			# print (chrom_list[c], "save_mem", save_mem)
			size = chrom_start_end[c, 1] - chrom_start_end[c, 0]

			# if size >= 1000:
			# 	pca_flag = True

			cell_size = int(math.ceil(size / scale_factor))

			data_within_chrom_list.append(np.copy(temp))
			weight_within_chrom_list.append(np.copy(temp_weight))

			per_cell_read = np.sum(temp_weight) / cell_num

			if save_mem:
				split_num = int(math.floor(len(temp) / 3e6))
				print (chrom_list[c], "split_num", split_num)
				cell_id = np.array_split(np.arange(cell_num), split_num)
				for part in range(split_num):
					mask = np.isin(temp[:, 0], cell_id[part])
					if not disable_mpl:
						p_list.append(
							pool.submit(create_matrix_one_chrom, config, c, size, cell_size, temp[mask], temp_weight[mask], chrom_start_end,
										cell_id[part], split_num, part, per_cell_read))
					else:
						p_list.append([config, c, size, cell_size, temp[mask], temp_weight[mask], chrom_start_end,
										cell_id[part], split_num, part, per_cell_read])
					
				cell_feats[c] = [[] for p in range(split_num)]
				qc_list[c] = [[] for p in range(split_num)]
				c2total_part_num[c] = split_num
			else:
				if not disable_mpl:
					p_list.append(
						pool.submit(create_matrix_one_chrom, config, c, size, cell_size, temp, temp_weight, chrom_start_end, cell_num, 1, 0, per_cell_read))
				else:
					p_list.append([config, c, size, cell_size, temp, temp_weight, chrom_start_end, cell_num, 1, 0, per_cell_read])
				cell_feats[c] = [[]]
				qc_list[c] = [[]]
				c2total_part_num[c] = 1
			temp_mask = temp
			temp_weight_mask = temp_weight
			
			def pseudo_bulk():
				bin_adj = csr_matrix((temp_weight_mask, (
				temp_mask[:, 2] - chrom_start_end[c, 0], temp_mask[:, 3] - chrom_start_end[c, 0])), shape=(size, size), dtype='float32')
				bin_adj = np.array(bin_adj.todense())
				bin_adj = bin_adj + bin_adj.T + np.diag(np.sum(bin_adj, axis=-1) == 0) #- np.diag(np.diagonal(bin_adj))
				mean_, std_ = np.mean(bin_adj), np.std(bin_adj)
				np.clip(bin_adj, a_min=None, a_max=mean_ + 10 * std_, out=bin_adj)
				bin_adj = normalize(bin_adj, axis=1, norm='l1')
				# bin_adj = sqrt_norm(bin_adj)
				# bin_adj = bin_adj / np.sum(bin_adj) * bin_adj.shape[0]
				return bin_adj
			
			if "bulk_path" not in config:
				bin_adj = pseudo_bulk()
			else:
				print ("using bulk hic")
				bk_path = config['bulk_path']
				if 'mcool' in bk_path:
					bk_path = bk_path+"::resolutions/%d" % res
				
				if 'cool' in bk_path:
					f = cooler.Cooler(bk_path)
					bin_adj = f.matrix(balance=False).fetch(chrom_list[c])
					bin_adj = np.array(bin_adj)
					bin_adj[np.isnan(bin_adj)] = 0.0
					bin_adj = bin_adj / np.sum(bin_adj) * bin_adj.shape[0]
					
				elif 'npy' in bk_path:
					bin_adj = np.load(bk_path)
					
				if len(bin_adj) != size:
					print("incorrect shape", "receives", bin_adj.shape, "should be",size)
					print ("fallback to pseudobulk")
					bin_adj = pseudo_bulk()
			binadj_dict[c] = bin_adj
			if size >= 3000:
				# scale
				sf = int(round(size / 3000))
				conv_filter = torch.ones(1, 1, 1, sf)#.to(device)
				conv_filter = conv_filter / torch.sum(conv_filter)
				B = F.conv2d(torch.from_numpy(bin_adj)[None, None, :, :].float(), conv_filter, stride=[1, sf])
				# print (B.shape, bin_adj.shape)
				bin_adj = B.detach().cpu().numpy()[0, 0, :, :]
			# if pca_flag:
			# 	size1 = int(0.2 * len(bin_adj))
			# 	U, s, Vt = pca(bin_adj, k=size1, raw=True)  # Automatically centers.
			# 	bin_adj = np.array(U[:, :size1] * s[:size1])
			create_or_overwrite(save_file, "%d" % c, bin_adj)

		bar = trange(len(p_list), desc='creating matrices tasks')
		if not disable_mpl:
			for p in as_completed(p_list):
				chrom_count, c, a, b, total_part_num, part_num, cell_id, qc = p.result()
				total_reads[cell_id] += a.reshape((-1))
				total_possible += float(b) / total_part_num
				cell_feats[c][part_num] = chrom_count
				qc_list[c][part_num] = qc
				bar.update(1)
		else:
			for p in p_list:
				chrom_count, c, a, b, total_part_num, part_num, cell_id, qc = create_matrix_one_chrom(*p)
				total_reads[cell_id] += a.reshape((-1))
				total_possible += float(b) / total_part_num
				cell_feats[c][part_num] = chrom_count
				qc_list[c][part_num] = qc
				bar.update(1)

		bar.close()

		for c in range(len(chrom_list)):
			cell_feats[c] = np.concatenate(cell_feats[c])
			qc_list[c] = np.concatenate(qc_list[c])
			bin_adj = binadj_dict[c]
			size = np.sum(np.sum(bin_adj > 0, axis=-1) > 0.1 * len(bin_adj))
			qc_list[c] = np.asarray(qc_list[c])
			qc_list[c] = ((qc_list[c] >= max(0.25 * size - 5, 0))).astype('bool')
			
		pool.shutdown(wait=True)

		chrom2celladj = {}

		total_linear_chrom_size = 0.0

		for c in range(len(chrom_list)):
			total_part_num = c2total_part_num[c]
			if total_part_num == 1:
				non_diag_sparse_all = np.load(os.path.join(temp_dir, "temp", "sparse_gcn_%s.npy" % chrom_list[c]),
				                          allow_pickle=True)
				cell_adj_all = np.load(os.path.join(temp_dir, "temp", "cell_adj_%s.npy" % chrom_list[c]),
				                   allow_pickle=True)
			else:
				non_diag_sparse_all = []
				cell_adj_all = []
				origin_sparse_list = []
				for pt in range(total_part_num):
					non_diag_sparse_all.append(np.load(
						os.path.join(temp_dir, "temp", "sparse_gcn_%s_part_%d.npy" % (chrom_list[c], pt)),
						allow_pickle=True))
					cell_adj_all.append(np.load(
						os.path.join(temp_dir, "temp", "cell_adj_%s_part_%d.npy" % (chrom_list[c], pt)),
						allow_pickle=True))
					origin_sparse_list.append(np.load(os.path.join(temp_dir, "temp","%s_sparse_adj_part_%d.npy" % (chrom_list[c], pt)),
					                                  allow_pickle=True))

				non_diag_sparse_all = np.concatenate(non_diag_sparse_all)
				cell_adj_all = np.concatenate(cell_adj_all)
				origin_sparse_list = np.concatenate(origin_sparse_list)
				np.save(os.path.join(raw_dir, "%s_sparse_adj.npy" % chrom_list[c]), origin_sparse_list)


			sparse_chrom_list[c] = non_diag_sparse_all


			if "batch_id" in config:
				batch_id_info = fetch_batch_id(config, "batch_id")
			elif "library_id" in config:
				batch_id_info = fetch_batch_id(config, "library_id")
			else:
				batch_id_info = np.ones((cell_num))

			bulk = np.sum(cell_adj_all, axis=0) / len(cell_adj_all)

			bulk_bin = []
			for k in range(bulk.shape[0]):
				bulk_bin.append(np.sum(bulk[k, :]) / (bulk.shape[0]))
			bulk_bin = np.array(bulk_bin)

			batches = np.unique(batch_id_info)
			
			new_cell_adj_all1 = ["" for i in range(len(cell_adj_all))]
			new_cell_adj_all2 = ["" for i in range(len(cell_adj_all))]
			
			idx = np.triu_indices(cell_adj_all[0].shape[0], k=1)
			
			for index, b in enumerate(batches):
				b_bin = []
				b_c = np.sum(cell_adj_all[batch_id_info == b], axis=0) / np.sum(batch_id_info == b)
				for k in range(b_c.shape[0]):
					b_bin.append(np.sum(b_c[k, :]) / b_c.shape[0])
				b_bin = np.array(b_bin)
				
				if spearmanr(b_bin[bulk_bin > 0.0], bulk_bin[bulk_bin > 0.0])[0] < 0.8:
					print(c, "correct be for batch", b, spearmanr(b_bin[bulk_bin > 0.0], bulk_bin[bulk_bin > 0.0]))
					for i in np.where(batch_id_info == b)[0]:
						m = cell_adj_all[i]
						row_sums = b_bin + 1e-15
						row_indices, col_indices = m.nonzero()
						m.data /= row_sums[row_indices]
						m.data *= bulk_bin[row_indices]
						m = m / np.sum(m)
						m_diag = m.diagonal()
						m_nodiag = m - diags(m.diagonal())
						new_cell_adj_all1[i] = csr_matrix(m_nodiag[idx])
						new_cell_adj_all2[i] = csr_matrix(m_diag.reshape((1, -1)))
				else:
					for i in np.where(batch_id_info == b)[0]:
						m = cell_adj_all[i]
						# cell_adj_all[i] = m.reshape((1, -1))
						m_diag = m.diagonal()
						m_nodiag = m - diags(m.diagonal())
						new_cell_adj_all1[i] = csr_matrix(m_nodiag[idx])
						new_cell_adj_all2[i] = csr_matrix(m_diag.reshape((1, -1)))
			cell_adj_all = [vstack(new_cell_adj_all1).tocsr(), vstack(new_cell_adj_all2).tocsr()]
			
			chrom2celladj[c] = cell_adj_all
			total_linear_chrom_size += int(math.sqrt(list(cell_adj_all[0].shape)[-1]) * res_cell / 1000000)
		# print (total_linear_chrom_size)
		# pool = ProcessPoolExecutor(max_workers=cpu_num)
		if len(chrom_list) > 1:
			total_embed_size = min(max(int(cell_adj_all[0].shape[0] * 0.5), int(total_linear_chrom_size * 0.5)),
			                       int(cell_adj_all[0].shape[0] * 0.65))
		else:
			total_embed_size = int(np.min(cell_adj_all[0].shape) * 0.8)
		total_embed_size = min(total_embed_size, 2400)
		print("total_feats_size", total_embed_size)
		# p_list = []
		
		bar = trange(len(chrom_list))
		if "cell" not in save_file.keys():
			save_file_cell = save_file.create_group("cell")
		else:
			save_file_cell = save_file["cell"]
		
		# for p in as_completed(p_list):
		# 	temp1, c = p.result()


		for c in range(len(chrom_list)):
			temp = chrom2celladj[c]
			length = int(np.sqrt(temp[0].shape[-1]) / 1000000 * res_cell)
			size = int(total_embed_size / total_linear_chrom_size * length) + 1
			temp1, c = generate_feats_one(temp[0], temp[1], size, length, c, qc_list[c])
			bar.update(1)
			create_or_overwrite(save_file_cell, "%d" % c, data=temp1)

		bar.close()
		pool.shutdown(wait=True)

		total_sparsity = total_reads / total_possible
		# print("sparsity", total_sparsity.shape, total_sparsity, np.median(total_sparsity))

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
		# print (num)
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


		for c in range(len(chrom_list)):
			mask = (chrom_info == c) & ((data[:, 2] - data[:, 1]) >= 2)
			index = np.arange(int(np.sum(mask)))
			np.random.shuffle(index)
			train_index = index[:int(0.85 * len(index))]
			test_index = index[int(0.85 * len(index)):]
			create_or_overwrite(save_file, "train_data_%s" % chrom_list[c], data=data[mask][train_index] + 1)
			create_or_overwrite(save_file, "train_chrom_%s" % chrom_list[c], data=chrom_info[mask][train_index])
			create_or_overwrite(save_file, "train_weight_%s" % chrom_list[c], data=weight[mask][train_index].astype('float32'))

			create_or_overwrite(save_file, "test_data_%s" % chrom_list[c], data=data[mask][test_index] + 1)
			create_or_overwrite(save_file, "test_chrom_%s" % chrom_list[c], data=chrom_info[mask][test_index])
			create_or_overwrite(save_file, "test_weight_%s" % chrom_list[c], data=weight[mask][test_index].astype('float32'))

		# Add 1 for padding idx
		# create_or_overwrite(save_file, "data", data=data + 1)
		# create_or_overwrite(save_file, "chrom", data=chrom_info)
		# create_or_overwrite(save_file, "weight", data=weight)

		distance = data[:, 2] - data[:, 1]
		info = pd.DataFrame({'dis': distance, 'weight': weight, 'cell': data[:, 0]})
		info1 = info.groupby(by='dis').mean().reset_index()
		max_bin1 = int(np.max(num[2:]))
		distance2weight = np.zeros((max_bin1, 1), dtype='float32')
		distance2weight[np.array(info1['dis']), 0] = np.array(info1['weight'])
		create_or_overwrite(save_file, "distance2weight", data=distance2weight)

		info1 = info.groupby(by='cell').mean().reset_index()
		cell2weight = np.zeros((num[1], 1), dtype='float32')
		cell2weight[np.array(info1['cell']), 0] = np.array(info1['weight'])
		create_or_overwrite(save_file, "cell2weight", data=cell2weight)

		shutil.rmtree(os.path.join(temp_dir, "temp"))

		
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
def impute_all(config):
	get_free_gpu()
	# fetch info from config
	temp_dir = config['temp_dir']
	raw_dir = os.path.join(temp_dir, "raw")
	rw_dir = os.path.join(temp_dir, "rw")
	res = config['resolution']
	impute_list = config['impute_list']
	
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
	
	print("start conv random walk (scHiCluster) as baseline")
	sc_list = []
	with h5py.File(os.path.join(rw_dir, "schicluster.hdf5"), "w") as hicluster_f:
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
			# matrix_list = []
			# Minus one because the generate_binpair function would add one in the function
			samples = generate_binpair(0, a[0].shape[0], min_bin_, max_bin_) - 1
			with h5py.File(os.path.join(rw_dir, "rw_%s.hdf5" % c), "w") as f:
				f.create_dataset('coordinates', data = samples)
				for i, m in enumerate(tqdm(a)):
					m = np.array(m.todense())
					m = impute_gpu(np.array(m))
					v = m[samples[:, 0], samples[:, 1]]
					# matrix_list.append(m.reshape((-1)))
					f.create_dataset("cell_%d" % (i), data=v)
			# matrix_list = np.stack(matrix_list, axis=0)
			# model = PCA(n_components=int(min(matrix_list.shape) * 0.2) - 1)
			# matrix_list = model.fit_transform(matrix_list)
			# hicluster_f.create_dataset("%s_vec" % c, data=matrix_list)
			# hicluster_f.create_dataset("%s_var" % c, data=model.explained_variance_)


# generate feats for cell and bin nodes (one chromosome, multiprocessing)
def generate_feats_one(temp1, temp2, size, length, c, qc_list):
	# temp1 non-diag feats, temp2: diag feats
	if temp1.shape[0] > 3:
		results = []
		for temp in [temp1, temp2]:
			if len(temp.data) == 0:
				continue
			mask = np.array(np.sum(temp > 0, axis=0) > min(5, temp.shape[0] - 2))
			mask = mask.reshape((-1))
			temp = temp[:, mask]

			size = min(size, temp.shape[-1] - 2)
			mean_, std_ = np.mean(temp.data), np.std(temp.data)
			# print("mean", "std", "max", "0.9q", mean_, std_, np.max(temp.data), np.quantile(temp.data, 0.9))
			# print("# of outliers:", np.sum(temp.data > (mean_ + 15 * std_)), "# of nonzero terms", len(temp.data))
			np.clip(temp.data, a_min=None, a_max=mean_ + 10 * std_, out=temp.data)
			results.append(temp)
			
		if len(results) == 2:
			split = results[0].shape[1]
			temp = hstack(results)
			temp = normalize(temp, norm='l1', axis=1) * length
			temp1, temp2 = temp[:, :split], temp[:, split:]
		else:
			temp1, temp2 = temp, None
		# when there are more than 10 noisy cells, fit only using good cells
		# Otherwise, these noises are tolerated.
		if len(qc_list) - np.sum(qc_list) > 10:
			model = TruncatedSVD(n_components=size, algorithm='randomized', n_iter=2).fit(temp1[qc_list])
			temp1 = model.transform(temp1)
			if temp2 is not None:
				model = TruncatedSVD(n_components=size, algorithm='randomized', n_iter=2).fit(temp2[qc_list])
				temp2 = model.transform(temp2)
		else:
			temp1 = TruncatedSVD(n_components=size, algorithm='randomized', n_iter=2).fit_transform(temp1)
			if temp2 is not None:
				temp2 = TruncatedSVD(n_components=size, algorithm='randomized', n_iter=2).fit_transform(temp2)
		if temp2 is not None:
			temp1 = np.concatenate([temp1, temp2], axis=1)
	else:
		temp1 = np.eye(temp1.shape[0])
	
	return temp1, c


def check_sparsity(temp):

	total_reads, total_possible = np.array(np.sum(temp > 0, axis=-1)), temp.shape[1]
	return total_reads, total_possible
	
	
def process_signal_one(chrom):
	cmd = ["python", "Coassay_pretrain.py", args.config, chrom]
	subprocess.call(cmd)


def process_signal(config):
	# fetch info from data_dir
	data_dir = config['data_dir']
	chrom_list = config['chrom_list']
	temp_dir = config['temp_dir']
	gpu_num = config['gpu_num']
	
	if not os.path.exists(os.path.join(temp_dir, "temp")):
		os.mkdir(os.path.join(temp_dir, "temp"))
		
	
	print("co-assay mode")
	signal_file = h5py.File(os.path.join(data_dir, "sc_signal.hdf5"), "r")
	signal_names = config["coassay_signal"]
	chrom2signals = {chrom: [] for chrom in chrom_list}
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
		np.save(os.path.join(temp_dir, "temp", "coassay_%s.npy" % chrom), temp)
		signal_all.append(temp)
	signal_all = np.concatenate(signal_all, axis=-1)
	signal_all = PCA(n_components=int(np.min(signal_all.shape) * 0.8)).fit_transform(signal_all)
	np.save(os.path.join(temp_dir, "temp", "coassay_all.npy"), signal_all)
	
	
	

	pool = ProcessPoolExecutor(max_workers=int(gpu_num * 1.2))
	for chrom in chrom_list:
		pool.submit(process_signal_one, chrom)
		time.sleep(3)
	pool.shutdown(wait=True)
	

	
	attributes_list = []
	for chrom in chrom_list:
		temp = np.load(os.path.join(temp_dir, "temp", "pretrain_coassay_%s.npy" % chrom))
		attributes_list.append(temp)
		
		
		
	attributes_list = np.concatenate(attributes_list, axis=-1)
	attributes_list = StandardScaler().fit_transform(attributes_list)
	

	np.save(os.path.join(temp_dir, "pretrain_coassay.npy"), attributes_list)
	shutil.rmtree(os.path.join(temp_dir, "temp"))

def scool_rwr(config):
	chrom_list = config['impute_list']
	temp_dir = config['temp_dir']
	raw_dir = os.path.join(temp_dir, "raw")
	rw_dir = os.path.join(temp_dir, "rw")
	res = config['resolution']
	
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


def scool_raw(config):
	chrom_list = config['chrom_list']
	temp_dir = config['temp_dir']
	raw_dir = os.path.join(temp_dir, "raw")
	rw_dir = os.path.join(temp_dir, "rw")
	res = config['resolution']
	
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
		
		

if __name__ == '__main__':
	args = parse_args()
	config = get_config(args.config)
	
	create_dir(config)
	generate_chrom_start_end(config)
	extract_table(config)
	create_matrix(config)
	
	if "coassay" in config:
		if config["coassay"]:
			process_signal(config)

	if "random_walk" in config:
		if config["random_walk"]:
			impute_all(config)
	
	
	# scool_rwr(config)
	# scool_raw(config)