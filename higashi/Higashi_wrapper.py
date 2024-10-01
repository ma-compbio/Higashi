import multiprocessing as mp
import warnings
import torch.optim
try:
	from .Higashi_backend.Modules import *
	from .Higashi_backend.Functions import *
	from .Higashi_backend.utils import *
	from .Impute import impute_process
except:
	try:
		from Higashi_backend.Modules import *
		from Higashi_backend.Functions import *
		from Higashi_backend.utils import *
		from Impute import impute_process
	except:
		raise EOFError
import argparse
import resource
from scipy.sparse import csr_matrix
from scipy.sparse.csr import get_csr_submatrix
from sklearn.preprocessing import StandardScaler
import pickle
import subprocess
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
try:
	get_ipython()
	from tqdm.notebook import tqdm, trange
except:
	pass

torch.backends.cudnn.benchmark = True
torch.set_default_dtype(torch.float32)


def parse_args():
	parser = argparse.ArgumentParser(description="Higashi main program")
	parser.add_argument('-c', '--config', type=str, default="../config_dir/config_ramani.JSON")
	parser.add_argument('-s', '--start', type=int, default=1)
	parser.add_argument('-e', '--end', type=int, default=3)
	
	return parser.parse_args()


def get_free_gpu(num=1, change_cur=True):
	# os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > ./tmp')
	# memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
	os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Total > ./tmp1')
	os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used > ./tmp2')
	memory_all = [int(x.split()[2]) for x in open('tmp1', 'r').readlines()]
	memory_used = [int(x.split()[2]) for x in open('tmp2', 'r').readlines()]
	memory_available = [m1 - m2 for m1, m2 in zip(memory_all, memory_used)]
	if len(memory_available) > 0:
		max_mem = np.max(memory_available)
		if num == 1 and change_cur:
			ids = np.where(memory_available == max_mem)[0]
			chosen_id = int(np.random.choice(ids, 1)[0])
			print("setting to gpu:%d" % chosen_id)
			torch.cuda.set_device(chosen_id)
			return "cuda:%d" % chosen_id
		else:
			ids = np.argsort(memory_available)[::-1][:num]
			return ids
	
	else:
		return



def check_nonzero(x, c):
	# minus 1 because add padding index
	mtx = sparse_chrom_list_dict[c][x[0] - 1]
	M, N = mtx.shape
	if mem_efficient_flag:
		row_start = max(0, x[1] - 1 - num_list[c] - 1)
		row_end = min(x[1] - 1 - num_list[c] + 2, N - 1)
		col_start = max(0, x[2] - 1 - num_list[c] - 1)
		col_end = min(x[2] - 1 - num_list[c] + 2, N - 1)
	else:
		row_start = x[1] - 1 - num_list[c]
		row_end = min(row_start + 1, N - 1)
		col_start = x[2] - 1 - num_list[c]
		col_end = min(col_start + 1, N - 1)
	try:
		indptr, nbrs, nbr_value = get_csr_submatrix(
			M, N, mtx.indptr, mtx.indices, mtx.data, row_start, row_end, col_start, col_end)
	except:
		print(M, N, num_list[c], x[1], row_start, row_end, x[2], col_start, col_end, mem_efficient_flag)
	a = len(nbr_value) > 0
	
	return a


def generate_negative_cpu(x, x_chrom, neg_num, max_bin, forward=True):
	rg = np.random.default_rng()
	neg_list, neg_chrom = np.zeros((x.shape[0] * neg_num, x.shape[1]), dtype='int'), \
	                      np.zeros((x.shape[0] * neg_num), dtype='int8')
	
	if forward:
		func1 = pass_
	else:
		func1 = tqdm
	
	success_count = 0
	
	change_list_all = rg.integers(0, x.shape[-1], (len(x), neg_num))
	simple_or_hard_all = rg.random((len(x), neg_num))
	for j, sample in enumerate(func1(x)):
		
		for i in range(neg_num):
			temp = np.copy(sample)
			change = change_list_all[j, i]
			trial = 0
			while check_nonzero(temp, x_chrom[j]):
				# while flag:
				if steps == 1:
					temp = np.copy(sample)
				
				# Try too many times on one sample, move on
				trial += 1
				if trial >= 50:
					temp = ""
					break
				
				# hard mode, Only change one node
				if simple_or_hard_all[j, i] <= pair_ratio:
					start, end = start_end_dict[int(temp[change])]
					
					# It's changing the bin
					if change != 0:
						other_bin = 2 if change == 1 else 2
						other_bin = temp[other_bin]
						start = max(start, other_bin - max_bin)
						end = min(end, other_bin + max_bin)
						
						temp[change] = np.random.randint(
							int(start), int(end), 1) + 1
					else:
						temp[change] = rg.choice(end - start) + start + 1
				
				else:
					start, end = start_end_dict[int(temp[0])]
					
					temp[0] = rg.choice(end - start) + start + 1
					
					start, end = start_end_dict[int(temp[1])]
					
					temp[1] = rg.choice(end - start) + start + 1
					
					start = max(start, temp[1] - max_bin)
					end = min(end, temp[1] + max_bin)
					
					temp[2] = rg.choice(end - start) + start + 1
				
				temp.sort()
				
				# Not a suitable sample
				if ((temp[2] - temp[1]) >= max_bin) or ((temp[2] - temp[1]) <= 1):
					temp = np.copy(sample)
			
			if len(temp) > 0:
				neg_list[success_count, :] = temp
				neg_chrom[success_count] = x_chrom[j]
				success_count += 1
			if success_count == neg_num * len(x):
				break
	return neg_list[:success_count], neg_chrom[:success_count]


def to_neighs_to_mask(to_neighs):
	torch_has_csr = False
	samp_neighs = to_neighs.reshape((-1))
	unique_nodes = {}
	unique_nodes_list = []
	
	count = 0
	column_indices = []
	row_indices = []
	if torch_has_csr:
		crow_indices = [0]
	else:
		row_indices = []
	v = []
	
	for i, samp_neigh in enumerate(samp_neighs):
		if len(samp_neigh) == 0:
			continue
		
		w = samp_neigh[1]
		samp_neigh = samp_neigh[0]
		
		w /= np.sum(w)
		
		try:
			for n in samp_neigh:
				
				if n not in unique_nodes:
					unique_nodes[n] = count
					unique_nodes_list.append(n)
					count += 1
				column_indices.append(unique_nodes[n])
				
				if not torch_has_csr:
					row_indices.append(i)
			if torch_has_csr:
				crow_indices.append(crow_indices[-1] + len(samp_neigh))
		except:
			print(i, samp_neigh, samp_neighs[i])
			raise EOFError
		
		v.append(w)
	
	v = np.concatenate(v, axis=0)
	
	unique_nodes_list = torch.LongTensor(unique_nodes_list)
	
	if torch_has_csr:
		return (torch.from_numpy(np.asarray([crow_indices, column_indices])), torch.from_numpy(v), unique_nodes_list)
	else:
		return (torch.from_numpy(np.asarray([row_indices, column_indices])), torch.from_numpy(v), unique_nodes_list)


def sum_duplicates(col, data):
	order = np.argsort(col)
	col = col[order]
	data = data[order]
	unique_mask = (col[1:] != col[:-1])
	unique_mask = np.append(True, unique_mask)
	col = col[unique_mask]
	unique_inds, = np.nonzero(unique_mask)
	data = np.add.reduceat(data, unique_inds)
	return col, data


def one_thread_generate_neg(edges_part, edges_chrom, edge_weight,
                            collect_num=1, training=False, chroms_in_batch=None):
	# global sparse_chrom_list_GCN, neg_num
	if neg_num == 0:
		y = np.ones((len(edges_part), 1))
		w = np.ones((len(edges_part), 1)) * edge_weight.reshape((-1, 1))
		x = edges_part
	else:
		neg_list, neg_chrom = generate_negative_cpu(edges_part, edges_chrom, neg_num, max_bin, True)
		neg_list = np.array(neg_list)[: len(edges_part) * neg_num, :]
		neg_chrom = np.array(neg_chrom)[: len(edges_part) * neg_num]
		if len(neg_list) == 0:
			raise EOFError
		
		correction = 1.0 if mode == "classification" else 0.0
		y = np.concatenate([np.ones((len(edges_part), 1)),
		                    np.zeros((len(neg_list), 1))])
		w = np.concatenate([np.ones((len(edges_part), 1)) * edge_weight.reshape((-1, 1)),
		                    np.ones((len(neg_list), 1)) * correction])
		x = np.concatenate([edges_part, neg_list])
		x_chrom = np.concatenate([edges_chrom, neg_chrom])

	
	new_x, new_y, new_w, new_x_chrom = [], [], [], []
	
	if graphsagemode:
		cell_ids = np.stack([x[:, 0], x[:, 0]], axis=-1).reshape((-1))
		bin_ids = x[:, 1:].reshape((-1))
		remove_bin_ids = np.copy(x[:, 1:])
		remove_bin_ids = remove_bin_ids[:, ::-1]
		nodes_chrom = np.stack([x_chrom, x_chrom], axis=-1).reshape((-1))
		to_neighs = []
		
		rg = np.random.default_rng()
		remove_or_not = rg.random(len(nodes_chrom))
		
		for i, (c, cell_id, bin_id, remove_bin_id) in enumerate(
				zip(nodes_chrom, cell_ids, bin_ids, remove_bin_ids.reshape((-1)))):
			if precompute_weighted_nbr:
				mtx = sparse_chrom_list_GCN[c][cell_id - 1]
				row = bin_id - 1 - num_list[c]
				M, N = mtx.shape
				indptr, nbrs, nbr_value = get_csr_submatrix(
					M, N, mtx.indptr, mtx.indices, mtx.data, row, row + 1, 0, N)
			else:
				if weighted_adj:
					row_record, row_weight_record = [], []
					for nbr_cell in cell_neighbor_list[cell_id]:
						balance_weight = weight_dict[(nbr_cell, cell_id)]
						mtx = sparse_chrom_list_GCN[c][nbr_cell - 1]
						row = bin_id - 1 - num_list[c]
						M, N = mtx.shape
						indptr, nbrs_pt, nbr_value_pt = get_csr_submatrix(
							M, N, mtx.indptr, mtx.indices, mtx.data, row, row + 1, 0, N)
						row_record.append(nbrs_pt)
						row_weight_record.append(nbr_value_pt * balance_weight)

					nbrs, nbr_value = sum_duplicates(np.concatenate(row_record), np.concatenate(row_weight_record))
				else:
					mtx = sparse_chrom_list_GCN[c][cell_id - 1]
					row = bin_id - 1 - num_list[c]
					M, N = mtx.shape
					indptr, nbrs, nbr_value = get_csr_submatrix(
						M, N, mtx.indptr, mtx.indices, mtx.data, row, row + 1, 0, N)
			
			if training and (remove_or_not[i] >= 0.6):
				if len(nbrs) > 1:
					mask = nbrs != (remove_bin_id - 1 - num_list[c])
					nbrs = nbrs[mask]
					nbr_value = nbr_value[mask]
			
			nbr_value = np.log1p(nbr_value)
			nbrs = nbrs.reshape((-1)) + 1 + num_list[c]
			
			if type(nbrs) is not np.ndarray:
				print(row, nbrs)
			if len(nbrs) > 0:
				temp = [nbrs, nbr_value]
			else:
				temp = []
			to_neighs.append(temp)
		
		# Force to append an empty list and remove it, such that np.array won't broadcasting shapes
		to_neighs.append([])
		to_neighs = np.array(to_neighs, dtype='object')[:-1]
		to_neighs = np.array(to_neighs, dtype='object').reshape((len(x), 2))
		
		size = int(len(x) / collect_num)
		to_neighs_new = []
		
		if collect_num == 1:
			index = np.random.permutation(len(x))
			x, y, w, x_chrom, remove_bin_ids = x[index], \
			                                   y[index], \
			                                   w[index], \
			                                   x_chrom[index], \
			                                   remove_bin_ids[index]
			to_neighs = [to_neighs_to_mask(to_neighs[index])]
		else:
			
			for j in range(collect_num):
				x_part, y_part, w_part, x_chrom_part, \
				to_neighs_part, remove_bin_ids_part = x[j * size: min((j + 1) * size,
					                                                                            len(x))], \
				                                                                            y[j * size: min((j + 1) * size,
					                                                                                          len(x))], w[
				                                                                                                        j * size: min(
					                                                                                                        (
								                                                                                                        j + 1) * size,
					                                                                                                        len(x))], x_chrom[
				                                                                                                                      j * size: min(
					                                                                                                                      (
								                                                                                                                      j + 1) * size,
					                                                                                                                      len(x))], to_neighs[
				                                                                                                                                    j * size: min(
					                                                                                                                                    (
								                                                                                                                                    j + 1) * size,
					                                                                                                                                    len(x))], remove_bin_ids[
				                                                                                                                                                  j * size: min(
					                                                                                                                                                  (
								                                                                                                                                                  j + 1) * size,
					                                                                                                                                                  len(x))]
				
				index = np.random.permutation(len(x_part))
				
				x_part, y_part, w_part, x_chrom_part, to_neighs_part, remove_bin_ids_part = x_part[index], \
				                                                                            y_part[index], \
				                                                                            w_part[index], \
				                                                                            x_chrom_part[index], \
				                                                                            to_neighs_part[index], \
				                                                                            remove_bin_ids_part[index]
				new_x.append(x_part)
				new_y.append(y_part)
				new_w.append(w_part)
				new_x_chrom.append(x_chrom_part)
				to_neighs_new.append(to_neighs_to_mask(to_neighs_part))
			to_neighs = to_neighs_new
	else:
		size = int(len(x) / collect_num)
		if collect_num == 1:
			index = np.random.permutation(len(x))
			x, y, w, x_chrom = x[index], \
			                   y[index], \
			                   w[index], \
			                   x_chrom[index]
			to_neighs = [[]]
		else:
			for j in range(collect_num):
				x_part, y_part, w_part, x_chrom_part = x[
				                                       j * size: min((j + 1) * size, len(x))], y[
				                                                                               j * size: min(
					                                                                               (j + 1) * size,
					                                                                               len(x))], w[
				                                                                                             j * size: min(
					                                                                                             (
								                                                                                             j + 1) * size,
					                                                                                             len(x))], x_chrom[
				                                                                                                           j * size: min(
					                                                                                                           (
								                                                                                                           j + 1) * size,
					                                                                                                           len(x))]
				
				index = np.random.permutation(len(x_part))
				x_part, y_part, w_part, x_chrom_part = x_part[index], \
				                                       y_part[index], \
				                                       w_part[index], \
				                                       x_chrom_part[index]
				new_x.append(x_part)
				new_y.append(y_part)
				new_w.append(w_part)
				new_x_chrom.append(x_chrom_part)
			to_neighs = [[] for i in range(collect_num)]
			
			x = np.concatenate(new_x, axis=0)
			y = np.concatenate(new_y, axis=0)
			w = np.concatenate(new_w, axis=0)
			x_chrom = np.concatenate(new_x_chrom, axis=0)
	return x, y, w, x_chrom, to_neighs, chroms_in_batch



def mp_impute(config_path, path, name, mode, cell_start, cell_end, sparse_path, weighted_info=None, gpu_id=None):
	import os
	path1 = os.path.abspath(__file__)
	dir_path = os.path.dirname(path1)
	print(dir_path)
	impute_file_path = str(os.path.join(dir_path, "Impute.py"))
	print (impute_file_path)
	cmd = ["python", impute_file_path, config_path, path, name, mode, str(int(cell_start)), str(int(cell_end)), sparse_path]
	if weighted_info is not None:
		cmd += [weighted_info]
	else:
		cmd += ["None"]
	
	if gpu_id is not None:
		cmd += [str(int(gpu_id))]
	else:
		cmd += ["None"]
	print(cmd)
	subprocess.call(cmd)



class Higashi():
	def __init__(self, config_path):
		super().__init__()
		self.config_path = config_path
		self.config = get_config(config_path)
		try:
			from .Process import create_dir
		except:
			from Process import create_dir
			
		create_dir(self.config)
		warnings.filterwarnings("ignore")
		rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
		resource.setrlimit(resource.RLIMIT_NOFILE, (3600, rlimit[1]))
		
		
	# For processing data: old Process.py
	def process_data(self, disable_mpl=False):
		self.generate_chrom_start_end()
		self.extract_table()
		self.create_matrix(disable_mpl)
		try:
			from .Process import process_signal, impute_all
		except:
			from Process import process_signal, impute_all
			
		if "coassay" in self.config:
			if self.config["coassay"]:
				process_signal(self.config)
		
		if "random_walk" in self.config:
			if self.config["random_walk"]:
				impute_all(self.config)
	
	def generate_chrom_start_end(self):
		try:
			from .Process import generate_chrom_start_end
		except:
			from Process import generate_chrom_start_end
		generate_chrom_start_end(self.config)
	
	def extract_table(self):
		try:
			from .Process import extract_table
		except:
			from Process import extract_table
		extract_table(self.config)
	
	def create_matrix(self, disable_mpl=False):
		try:
			from .Process import create_matrix
		except:
			from Process import create_matrix
		create_matrix(self.config, disable_mpl)
	
	# fetch information from config.JSON
	def fetch_info_from_config(self):
		config = self.config
		
		self.cpu_num = config['cpu_num']
		if self.cpu_num < 0:
			self.cpu_num = int(mp.cpu_count())
		print ("cpu_num", self.cpu_num)
		self.gpu_num = config['gpu_num']
		
		if 'cpu_num_torch' in config:
			self.cpu_num_torch = config['cpu_num_torch']
			if self.cpu_num_torch < 0:
				self.cpu_num_torch = int(mp.cpu_count())
		else:
			self.cpu_num_torch = self.cpu_num
			
		if torch.cuda.is_available():
			self.current_device = get_free_gpu()
		else:
			self.current_device = 'cpu'
			torch.set_num_threads(self.cpu_num_torch)

		self.data_dir = config['data_dir']
		self.temp_dir = config['temp_dir']
		self.embed_dir = os.path.join(self.temp_dir, "embed")
		self.chrom_list = config['chrom_list']
		print("training on data from:", self.chrom_list)
		if self.gpu_num < 2:
			self.non_para_impute = True
		else:
			self.non_para_impute = False
		
		self.dimensions = config['dimensions']
		self.impute_list = config['impute_list']
		self.res = config['resolution']
		self.neighbor_num = config['neighbor_num'] + 1
		self.mode = config["loss_mode"]
		global mode
		mode = self.mode
		if self.mode == 'rank':
			self.rank_thres = config['rank_thres']
		self.embedding_name = config['embedding_name']
		
		if "coassay" in config:
			self.coassay = config['coassay']
		else:
			self.coassay = False

		if 'pre_cell_embed' in config:
			self.pre_cell_embed = config['pre_cell_embed']
		else:
			self.pre_cell_embed = False

		self.save_path = os.path.join(self.temp_dir, "model")
		if not os.path.exists(self.save_path):
			os.makedirs(self.save_path)
		self.save_path = os.path.join(self.save_path, 'model.chkpt')
		
		self.impute_no_nbr_flag = True
		if "impute_no_nbr" in config:
			self.impute_no_nbr_flag = config['impute_no_nbr']
		
		self.impute_with_nbr_flag = True
		if "impute_with_nbr" in config:
			self.impute_with_nbr_flag = config['impute_with_nbr']
		
		if "embedding_epoch" in config:
			self.embedding_epoch = config['embedding_epoch']
		else:
			self.embedding_epoch = 60
		
		if "no_nbr_epoch" in config:
			self.no_nbr_epoch = config['no_nbr_epoch']
		else:
			self.no_nbr_epoch = 45
		if "with_nbr_epoch" in config:
			self.with_nbr_epoch = config['with_nbr_epoch']
		else:
			self.with_nbr_epoch = 30
		
		self.remove_be_flag = False
		if "correct_be_impute" in config:
			if config['correct_be_impute']:
				self.remove_be_flag = True
		self.precompute_weighted_nbr = True
		if "precompute_weighted_nbr" in config:
			if not config["precompute_weighted_nbr"]:
				self.precompute_weighted_nbr = False
		global precompute_weighted_nbr
		precompute_weighted_nbr = self.precompute_weighted_nbr
		# Training related parameters, but these are hard coded as they usually don't require tuning
		# collect_num=x means, one cpu thread would collect x batches of samples
		self.collect_num = 1
		self.update_num_per_training_epoch = 1000
		self.update_num_per_eval_epoch = 10
		
		self.chrom_start_end = np.load(os.path.join(self.temp_dir, "chrom_start_end.npy"))
	
	# generating attributes for cell nodes and bin nodes
	def generate_attributes(self):
		embeddings = []
		targets = []
		cell_node_feats = []
		
		with h5py.File(os.path.join(self.temp_dir, "node_feats.hdf5"), "r") as save_file:
			
			for c in self.chrom_list:
				a = np.array(save_file["cell"]["%d" % self.chrom_list.index(c)])
				# a = np.eye(a.shape[0]).astype('float32')
				cell_node_feats.append(a)

			if self.pre_cell_embed:
				print ("pre_cell_embed")
				cell_node_feats = np.load(self.pre_cell_embed).astype('float32')
				targets.append(StandardScaler().fit_transform(cell_node_feats.reshape((-1, 1))).reshape((len(cell_node_feats), -1)))
				embeddings.append(StandardScaler().fit_transform(cell_node_feats.reshape((-1, 1))).reshape((len(cell_node_feats), -1)))

			elif self.coassay:
				print("coassay")
				cell_node_feats = np.load(os.path.join(self.temp_dir, "pretrain_coassay.npy")).astype('float32')
				targets.append(cell_node_feats)
				cell_node_feats = StandardScaler().fit_transform(cell_node_feats)
				embeddings.append(cell_node_feats)
			else:
				if self.num[0] >= 10:
					cell_node_feats1 = remove_BE_linear(cell_node_feats, self.config, self.data_dir, self.cell_feats1)
					cell_node_feats1 = StandardScaler().fit_transform(cell_node_feats1.reshape((-1, 1))).reshape(
						(len(cell_node_feats1), -1))
					cell_node_feats2 = [StandardScaler().fit_transform(x) for x in cell_node_feats]
					cell_node_feats2 = remove_BE_linear(cell_node_feats2, self.config, self.data_dir, self.cell_feats1)
					cell_node_feats2 = StandardScaler().fit_transform(cell_node_feats2.reshape((-1, 1))).reshape(
						(len(cell_node_feats2), -1))
				else:
					cell_node_feats1 = remove_BE_linear(cell_node_feats, self.config, self.data_dir, self.cell_feats1)
					cell_node_feats2 = cell_node_feats1

				targets.append(cell_node_feats2.astype('float32'))
				embeddings.append(cell_node_feats1.astype('float32'))
			
			for i, c in enumerate(self.chrom_list):
				temp = np.array(save_file["%d" % i]).astype('float32')
				# temp = np.eye(temp.shape[0]).astype('float32')
				# print ("min_max", np.min(temp), np.max(temp), np.median(temp))
				temp = StandardScaler().fit_transform(temp.reshape((-1, 1))).reshape((len(temp), -1))
				# print(np.min(temp), np.max(temp), np.median(temp))
				chrom = np.zeros((len(temp), len(self.chrom_list))).astype('float32')
				chrom[:, i] = 1
				list1 = [temp, chrom]
				
				temp = np.concatenate(list1, axis=-1)
				embeddings.append(temp)
				targets.append(temp)


		print("start making attribute")
		attribute_all = []
		for i in range(len(self.num) - 1):
			chrom = np.zeros((self.num[i + 1], len(self.chrom_list)))
			chrom[:, i] = 1
			coor = np.arange(self.num[i + 1]).reshape((-1, 1)).astype('float32')
			coor /= self.num[1]
			attribute = np.concatenate([chrom, coor], axis=-1)
			# attribute = chrom
			attribute_all.append(attribute)
		
		attribute_all = np.concatenate(attribute_all, axis=0)
		attribute_dict = np.concatenate([np.zeros((self.num[0] + 1, attribute_all.shape[-1])), attribute_all],
		                                axis=0).astype(
			'float32')
		
		return embeddings, attribute_dict, targets
	
	
	# Prepare the model for training and imputation
	def prep_model(self):
		self.fetch_info_from_config()
		global pair_ratio, weighted_adj, num, num_list, start_end_dict, mem_efficient_flag
		global neg_num, max_bin, mode, graphsagemode, precompute_weighted_nbr, weighted_adj
		global cell_neighbor_list, cell_neighbor_weight_list
		global sparse_chrom_list_GCN, sparse_chrom_list_dict, sparse_chrom_list
		
		weighted_adj = False
		self.cell_embeddings = None
		self.ori_sparse_list = {}

		# Load everything from the hdf5 file
		with h5py.File(os.path.join(self.temp_dir, "node_feats.hdf5"), "r") as input_f:
			num = np.array(input_f['num'])
			self.num = num
			train_data, train_weight, train_chrom = [], [], []
			test_data, test_weight, test_chrom = [], [], []
			
			for c in range(len(self.chrom_list)):
				train_data.append(np.array(input_f['train_data_%s' % self.chrom_list[c]]).astype('int'))
				train_weight.append(np.array(input_f['train_weight_%s' % self.chrom_list[c]]).astype('float32'))
				train_chrom.append(np.array(input_f['train_chrom_%s' % self.chrom_list[c]]).astype('int8'))
				
				test_data.append(np.array(input_f['test_data_%s' % self.chrom_list[c]]).astype('int'))
				test_weight.append(np.array(input_f['test_weight_%s' % self.chrom_list[c]]).astype('float32'))
				test_chrom.append(np.array(input_f['test_chrom_%s' % self.chrom_list[c]]).astype('int8'))
			
			self.distance2weight = np.array(input_f['distance2weight'])
			self.distance2weight = self.distance2weight.reshape((-1, 1))
			self.distance2weight = torch.from_numpy(self.distance2weight).float().to(device, non_blocking=True)
			
			self.total_sparsity_cell = np.array(input_f['sparsity'])
			# Trust the original cell feature more if there are more non-zero entries for faster convergence.
			# If there are more than 95% zero entries at 1Mb, then rely more on Higashi non-linear transformation
			self.median_total_sparsity_cell = np.median(self.total_sparsity_cell)
			print("total_sparsity_cell", self.median_total_sparsity_cell)
			if self.coassay or self.pre_cell_embed:
				print("contractive loss")
				self.contractive_flag = True
				self.contractive_loss_weight = 1e-3

			else:
				if self.median_total_sparsity_cell >= 0.05:
					print("contractive loss")
					self.contractive_flag = True
					self.contractive_loss_weight = 1e-3
				else:
					print("no contractive loss")
					self.contractive_flag = False
					self.contractive_loss_weight = 0.0

			start_end_dict = np.array(input_f['start_end_dict'])
			self.cell_feats = np.array(input_f['extra_cell_feats'])
			self.cell_feats1 = np.array(input_f['cell2weight'])
			self.cell_num = num[0]
			self.cell_ids = (torch.arange(self.num[0])).long().to(device, non_blocking=True)
			
		
		# automatically set batch size based on the resolution and number of cells
		self.batch_size = min(int(256 * max((1000000 / self.res), 1) * max(self.num[0] / 6000, 1)), 1280)
		print ("batch_size", self.batch_size)
		num_list = np.cumsum(self.num)
		self.num_list = num_list
		max_bin = int(np.max(self.num[1:]))
		# mem_efficient_flag = self.cell_num > 30
		mem_efficient_flag = True
		
		total_possible = 0
		
		for c in self.chrom_start_end:
			start, end = c[0], c[1]
			for i in range(start, end):
				for j in range(i, end):
					total_possible += 1
		
		total_possible *= self.cell_num
		sparsity = np.sum([len(d) + len(d2) for d, d2 in zip(train_data, test_data)]) / total_possible
		
		# Dependes on the sparsity, change the number of negative samples
		if sparsity > 0.3:
			neg_num = 1
		elif sparsity > 0.2:
			neg_num = 2
		elif sparsity > 0.15:
			neg_num = 3
		elif sparsity > 0.1:
			neg_num = 4
		else:
			neg_num = 5
		
		self.batch_size *= (1 + neg_num)
		print("Node type num", self.num, num_list)
		start_end_dict = np.concatenate([np.zeros((1, 2)), start_end_dict], axis=0).astype('int')
		
		self.cell_feats = np.sum(self.cell_feats, axis=-1, keepdims=True)
		self.cell_feats1 = torch.from_numpy(self.cell_feats1).float().to(device, non_blocking=True)
		self.cell_feats = np.log1p(self.cell_feats)
		self.label_info = pickle.load(open(os.path.join(self.data_dir, "label_info.pickle"), "rb"))
		if "batch_id" in self.config or "library_id" in self.config:
			if "batch_id" in self.config:
				label = np.array(self.label_info[self.config["batch_id"]])
			else:
				label = np.array(self.label_info[self.config["library_id"]])
			uniques = np.unique(label)
			target2int = np.zeros((len(label), len(uniques)), dtype='float32')
			
			for i, t in enumerate(uniques):
				target2int[label == t, i] = 1
			print(target2int.shape)
			self.cell_feats = np.concatenate([self.cell_feats, target2int], axis=-1)
		self.cell_feats = np.concatenate([np.zeros((1, self.cell_feats.shape[-1])), self.cell_feats], axis=0).astype(
			'float32')
		
		embeddings_initial, attribute_dict, targets_initial = self.generate_attributes()
		
		sparse_chrom_list = np.load(os.path.join(self.temp_dir, "sparse_nondiag_adj_nbr_1.npy"), allow_pickle=True)
		filesize = os.path.getsize(os.path.join(self.temp_dir, "sparse_nondiag_adj_nbr_1.npy"))
		filesize_in_GB = filesize / (1024*1024*1024)
		if filesize_in_GB > 20:
			self.precompute_weighted_nbr = False
			global precompute_weighted_nbr
			precompute_weighted_nbr = self.precompute_weighted_nbr
		
		for i in range(len(sparse_chrom_list)):
			for j in range(len(sparse_chrom_list[i])):
				sparse_chrom_list[i][j] = sparse_chrom_list[i][j].astype('float32')
		
		if not mem_efficient_flag:
			sparse_chrom_list_GCN = sparse_chrom_list
			sparse_chrom_list_dict = copy.deepcopy(sparse_chrom_list)
			conv_weight = torch.ones((1, 1, 3, 3)).float()
			for chrom in range(len(sparse_chrom_list)):
				for cell in range(len(sparse_chrom_list[0])):
					sparse_chrom_list_dict[chrom][cell] = np.array(sparse_chrom_list_dict[chrom][cell].todense())
					m = sparse_chrom_list_dict[chrom][cell][None, None, :, :]
					m = torch.from_numpy(m).float()
					m = F.conv2d(m, conv_weight, padding=2)
					sparse_chrom_list_dict[chrom][cell] = m.detach().cpu().numpy()[0, 0, 1:-1, 1:-1]
		else:
			sparse_chrom_list_GCN = sparse_chrom_list
			sparse_chrom_list_dict = sparse_chrom_list
		
		if self.mode == 'classification':
			train_weight_mean = np.mean(train_weight)
			
			train_weight, test_weight = transform_weight_class(train_weight, train_weight_mean, neg_num), \
			                            transform_weight_class(test_weight, train_weight_mean, neg_num)
		elif self.mode == 'rank':
			train_weight = [x + 1 for x in train_weight]
			test_weight = [x + 1 for x in test_weight]
		
		
		# Constructing the model
		self.node_embedding_init = MultipleEmbedding(
			embeddings_initial,
			self.dimensions,
			False,
			num_list, targets_initial).to(device, non_blocking=True)
		
		self.node_embedding_init.wstack[0].fit(embeddings_initial[0], 300,
		                                       sparse=False,
		                                       targets=torch.from_numpy(targets_initial[0]
		                                                                ).float().to(device, non_blocking=True),
		                                       batch_size=1024)
		
		self.higashi_model = Hyper_SAGNN(
			n_head=8,
			d_model=self.dimensions,
			d_k=16,
			d_v=16,
			diag_mask=True,
			bottle_neck=self.dimensions,
			attribute_dict=attribute_dict,
			cell_feats=self.cell_feats,
			encoder_dynamic_nn=self.node_embedding_init,
			encoder_static_nn=self.node_embedding_init,
			chrom_num=len(self.chrom_list)).to(device, non_blocking=True)
		
		
		
		cell_neighbor_list = [[i] for i in range(self.num[0] + 1)]
		cell_neighbor_weight_list = [[1] for i in range(self.num[0] + 1)]
		
		# construct the dataloader
		self.training_data_generator = DataGenerator(train_data, train_chrom, train_weight,
		                                             int(self.batch_size / (neg_num + 1) * self.collect_num),
		                                             True, num_list, k=self.collect_num)
		self.validation_data_generator = DataGenerator(test_data, test_chrom, test_weight,
		                                               int(self.batch_size / (neg_num + 1)),
		                                               False, num_list, k=1)


	def forward_batch_hyperedge(self, batch_data, batch_weight, batch_chrom, batch_to_neighs, y,
	                            chroms_in_batch):
		model = self.higashi_model
		x = batch_data
		w = batch_weight
		# plus one, because chr1 - id 0 - NN1
		pred, pred_var, pred_proba = model(x, (batch_chrom, batch_to_neighs), chroms_in_batch=chroms_in_batch + 1)
		
		if self.use_recon:
			adj = self.node_embedding_init.embeddings[0](self.cell_ids).float()
			targets = self.node_embedding_init.targets[0](self.cell_ids).float()
			embed, recon = self.node_embedding_init.wstack[0](adj, return_recon=True)
			mse_loss = F.mse_loss(recon, targets)
		else:
			mse_loss = torch.as_tensor([0], dtype=torch.float).to(device, non_blocking=True)
		
		if self.mode == 'classification':
			main_loss = F.binary_cross_entropy_with_logits(pred, y, weight=w)
		
		elif self.mode == 'rank':
			pred = F.softplus(pred)
			diff = (pred.view(-1, 1) - pred.view(1, -1)).view(-1)
			diff_w = (w.view(-1, 1) - w.view(1, -1)).view(-1)
			mask_rank = torch.abs(diff_w) > self.rank_thres
			diff = diff[mask_rank].float()
			diff_w = diff_w[mask_rank]
			label = (diff_w > 0).float()
			main_loss = F.binary_cross_entropy_with_logits(diff, label)
			
			if not self.use_recon:
				if neg_num > 0:
					mask_w_eq_zero = w == 0
					makeitzero = F.mse_loss(w[mask_w_eq_zero].float(), pred[mask_w_eq_zero].float())
					mse_loss += makeitzero
		
		elif self.mode == 'zinb':
			pred = F.softplus(pred)
			extra = (self.cell_feats1[batch_data[:, 0] - 1]).view(-1, 1)
			pred = pred * extra
			pred_var = F.softplus(pred_var)
			
			pred = torch.clamp(pred, min=1e-8, max=1e8)
			pred_var = torch.clamp(pred_var, min=1e-8, max=1e8)
			main_loss = -log_zinb_positive(w.float(), pred.float(), pred_var.float(), pred_proba.float())
			main_loss = main_loss.mean()
		elif self.mode == 'regression':
			pred = pred.float().view(-1)
			w = w.float().view(-1)
			main_loss = F.mse_loss(pred, w)
		else:
			print("wrong mode")
			raise EOFError
		return pred, main_loss, mse_loss
	
	
	def train_epoch(self, training_data_generator, optimizer_list, train_pool, train_p_list):
		model = self.higashi_model
		model.train()
		
		bce_total_loss = 0
		mse_total_loss = 0
		final_batch_num = 0
		
		batch_num = int(self.update_num_per_training_epoch / self.collect_num)
		y_list, w_list, pred_list = [], [], []
		bar = trange(batch_num * self.collect_num, desc=' - (Training) ', leave=False, )
		
		while len(train_p_list) < batch_num:
			edges_part, edges_chrom, edge_weight_part, chroms_in_batch = training_data_generator.next_iter()
			train_p_list.append(
				train_pool.submit(one_thread_generate_neg, edges_part, edges_chrom, edge_weight_part,
				                  self.collect_num, True,
				                  chroms_in_batch))
			
		finish_count = 0
		for p in as_completed(train_p_list):
			batch_edge_big, batch_y_big, batch_edge_weight_big, batch_chrom_big, batch_to_neighs_big, chroms_in_batch = p.result()
			batch_edge_big = np2tensor_hyper(batch_edge_big, dtype=torch.long)
			batch_y_big, batch_edge_weight_big = torch.from_numpy(batch_y_big), torch.from_numpy(batch_edge_weight_big)
			batch_edge_big, batch_y_big, batch_edge_weight_big = batch_edge_big.to(device,
			                                                                       non_blocking=True), batch_y_big.to(
				device, non_blocking=True), batch_edge_weight_big.to(device, non_blocking=True)
			size = int(len(batch_edge_big) / self.collect_num)
			for j in range(self.collect_num):
				batch_edge, batch_edge_weight, batch_y, batch_chrom, batch_to_neighs = batch_edge_big[
                                                                       j * size: min((j + 1) * size,
                                                                                     len(batch_edge_big))], \
                                                                       batch_edge_weight_big[
                                                                       j * size: min((j + 1) * size,
                                                                                     len(batch_edge_big))], \
                                                                       batch_y_big[
                                                                       j * size: min((j + 1) * size,
                                                                                     len(batch_edge_big))], \
                                                                       batch_chrom_big[
                                                                       j * size: min((j + 1) * size,
                                                                                     len(batch_edge_big))], \
                                                                       batch_to_neighs_big[j]

				pred, loss_bce, loss_mse = self.forward_batch_hyperedge(batch_edge,
				                                                   batch_edge_weight, batch_chrom,
				                                                   batch_to_neighs, y=batch_y,
				                                                   chroms_in_batch=chroms_in_batch)

				y_list.append(batch_y.detach().cpu())
				w_list.append(batch_edge_weight.detach().cpu())
				pred_list.append(pred.detach().cpu())

				final_batch_num += 1

				if self.use_recon:
					for opt in optimizer_list:
						opt.zero_grad(set_to_none=True)
					loss_bce.backward(retain_graph=True)
					try:
						main_norm = self.node_embedding_init.wstack[0].weight_list[0].grad.data.norm(2)
					except:
						main_norm = 0.0

					for opt in optimizer_list:
						opt.zero_grad(set_to_none=True)
					loss_mse.backward(retain_graph=True)

					recon_norm = self.node_embedding_init.wstack[0].weight_list[0].grad.data.norm(2)
					shape = self.node_embedding_init.wstack[0].weight_list[0].shape[1]
					ratio = self.beta * main_norm / recon_norm
					ratio1 = max(ratio, 100 * self.median_total_sparsity_cell - 3)

					if self.contractive_flag:
						contractive_loss = 0.0
						for i in range(len(self.node_embedding_init.wstack[0].weight_list)):
							contractive_loss += torch.sum(self.node_embedding_init.wstack[0].weight_list[i] ** 2)
							contractive_loss += torch.sum(self.node_embedding_init.wstack[0].reverse_weight_list[i] ** 2)

					else:
						contractive_loss = 0.0

				else:
					contractive_loss = 0.0
					ratio = 0.0
					ratio1 = 0.0

				train_loss = self.alpha * loss_bce + ratio1 * loss_mse + self.contractive_loss_weight * contractive_loss
				for opt in optimizer_list:
					opt.zero_grad(set_to_none=True)
				# backward
				train_loss.backward()

				# update parameters
				for opt in optimizer_list:
					opt.step()

				bar.update(n=1)
				bar.set_description("- (Train) BCE: %.3f MSE: %.3f norm_ratio: %.2f" %
				                    (loss_bce.item(), loss_mse.item(), ratio1),
				                    refresh=False)

				bce_total_loss += loss_bce.item()
				mse_total_loss += loss_mse.item()
			
			train_p_list.remove(p)
			
			while len(train_p_list) < batch_num:
				edges_part, edges_chrom, edge_weight_part, chroms_in_batch = training_data_generator.next_iter()
				train_p_list.append(
					train_pool.submit(one_thread_generate_neg, edges_part, edges_chrom, edge_weight_part,
					                  self.collect_num,
					                  True,
					                  chroms_in_batch))
			
			finish_count += 1
			if finish_count == batch_num:
				break
		
		y = torch.cat(y_list)
		w = torch.cat(w_list)
		pred = torch.cat(pred_list)
		
		auc1, auc2, str1, str2 = roc_auc_cuda(w, pred)
		bar.close()
		return bce_total_loss / final_batch_num, mse_total_loss / final_batch_num, accuracy(
			y.view(-1), pred.view(-1)), auc1, auc2, str1, str2, train_pool, train_p_list
	
	
	def eval_epoch(self, validation_data_generator, p_list=None):
		"""Epoch operation in evaluation phase"""
		bce_total_loss = 0
		model = self.higashi_model
		model.eval()
		with torch.no_grad():
			y_list, w_list, pred_list = [], [], []
			
			bar = tqdm(range(self.update_num_per_eval_epoch), desc='  - (Validation)   ', leave=False)
			if p_list is None:
				pool = ProcessPoolExecutor(max_workers=self.cpu_num)
				p_list = []
				
				for i in range(self.update_num_per_eval_epoch):
					edges_part, edges_chrom, edge_weight_part, chroms_in_batch = validation_data_generator.next_iter()
					p_list.append(
						pool.submit(one_thread_generate_neg, edges_part, edges_chrom, edge_weight_part,
						            1, False,
						            chroms_in_batch))
			
			for p in as_completed(p_list):
				batch_x, batch_y, batch_w, batch_chrom, batch_to_neighs, chroms_in_batch = p.result()
				batch_x = np2tensor_hyper(batch_x, dtype=torch.long)
				batch_y, batch_w = torch.from_numpy(batch_y), torch.from_numpy(batch_w)
				batch_x, batch_y, batch_w = batch_x.to(device, non_blocking=True), batch_y.to(device,
				                                                                              non_blocking=True), batch_w.to(
					device, non_blocking=True)
				
				pred_batch, eval_loss, eval_mse = self.forward_batch_hyperedge(batch_x, batch_w,
				                                                          batch_chrom,
				                                                          batch_to_neighs[0], y=batch_y,
				                                                          chroms_in_batch=chroms_in_batch)
				
				bce_total_loss += eval_loss.item()
				
				y_list.append(batch_y.detach().cpu())
				w_list.append(batch_w.detach().cpu())
				pred_list.append(pred_batch.detach().cpu())
				
				bar.update(n=1)
				bar.set_description("- (Valid) BCE: %.3f MSE: %.3f " %
				                    (eval_loss.item(), eval_mse.item()),
				                    refresh=False)
			
			bar.close()
			
			y = torch.cat(y_list)
			w = torch.cat(w_list)
			pred = torch.cat(pred_list)
			
			auc1, auc2, str1, str2 = roc_auc_cuda(w, pred)
		
		return bce_total_loss / (len(p_list)), accuracy(y.view(-1), pred.view(-1)), auc1, auc2, str1, str2
	
	
	def train(self, training_data_generator, validation_data_generator, optimizer, epochs, load_first,
	          save_embed=False, save_name=""):
		model = self.higashi_model
		global pair_ratio, neg_num, max_bin, mode, graphsagemode, precompute_weighted_nbr, weighted_adj, print_str
		print_str = ""
		graphsagemode, precompute_weighted_nbr = isinstance(model.encode1.dynamic_nn, GraphSageEncoder_with_weights), \
		                                                        self.precompute_weighted_nbr
		no_improve = 0
		if load_first:
			checkpoint = torch.load(self.save_path + save_name)
			model.load_state_dict(checkpoint['model_link'])
		
		best_train_loss = 1000


		if save_embed:
			self.save_embeddings()
		
		eval_pool = ProcessPoolExecutor(max_workers=self.cpu_num)
		
		train_pool = ProcessPoolExecutor(max_workers=self.cpu_num)
		train_p_list = []
		
		for epoch_i in range(epochs):
			if save_embed:
				self.save_embeddings()
			
			print('[ Epoch', epoch_i, 'of', epochs, ']')
			eval_p_list = []
			for i in range(self.update_num_per_eval_epoch):
				edges_part, edges_chrom, edge_weight_part, chroms_in_batch = validation_data_generator.next_iter()
				eval_p_list.append(
					eval_pool.submit(one_thread_generate_neg, edges_part, edges_chrom, edge_weight_part,
					                 1, False,
					                 chroms_in_batch))
			
			start = time.time()
			
			bce_loss, mse_loss, train_accu, auc1, auc2, str1, str2, train_pool, train_p_list = self.train_epoch(
				training_data_generator, optimizer, train_pool, train_p_list)
			print('- (Train)   bce: {bce_loss: 7.4f}, mse: {mse_loss: 7.4f}, '
			      ' acc: {accu:3.3f} %, {str1}: {auc1:3.3f}, {str2}: {auc2:3.3f}, '
			      'elapse: {elapse:3.3f} s'.format(
				bce_loss=bce_loss,
				mse_loss=mse_loss,
				accu=100 *
				     train_accu,
				str1=str1,
				auc1=auc1,
				str2=str2,
				auc2=auc2,
				elapse=(time.time() - start)))
			
			start = time.time()
			valid_bce_loss, valid_accu, valid_auc1, valid_auc2, str1, str2 = self.eval_epoch(
				validation_data_generator, eval_p_list)
			print('- (Valid) bce: {bce_loss: 7.4f},'
			      '  acc: {accu:3.3f} %,'
			      '{str1}: {auc1:3.3f}, {str2}: {auc2:3.3f},'
			      'elapse: {elapse:3.3f} s'.format(
				bce_loss=valid_bce_loss,
				accu=100 *
				     valid_accu,
				str1=str1,
				auc1=valid_auc1,
				auc2=valid_auc2,
				str2=str2,
				elapse=(time.time() - start)))
			
			# Dynamic pair ratio for stage one
			if self.dynamic_pair_ratio:
				if pair_ratio < 0.5:
					pair_ratio += 0.1
					print_str += "pair_ratio: %.1f\t" % pair_ratio
				else:
					pair_ratio = 0.5
				# scheduler.step(bce_loss)
				
			
			if not self.dynamic_pair_ratio:
				if bce_loss < best_train_loss - 1e-3:
					best_train_loss = bce_loss
					no_improve = 0
				else:
					no_improve += 1
			
			# scheduler.step(bce_loss)
			
			if no_improve >= 4:
				print_str += "no improvement early stopping\t"
				break
			if no_improve > 0:
				print_str += "no improve: %d\t" % no_improve
			
			if epoch_i % 5 == 0:
				checkpoint = {
					'model_link': model.state_dict(),
					'epoch': epoch_i}
				torch.save(checkpoint, self.save_path + save_name)
			print (print_str)
			print_str = ""
		start = time.time()
		
		valid_bce_loss, valid_accu, valid_auc1, valid_auc2, _, _ = self.eval_epoch(validation_data_generator)
		print('  - (Validation-hyper) bce: {bce_loss: 7.4f},'
		      '  acc: {accu:3.3f} %,'
		      ' auc: {auc1:3.3f}, aupr: {auc2:3.3f},'
		      'elapse: {elapse:3.3f} s'.format(
			bce_loss=valid_bce_loss,
			accu=100 *
			     valid_accu,
			auc1=valid_auc1,
			auc2=valid_auc2,
			elapse=(time.time() - start)))
		train_pool.shutdown()
		eval_pool.shutdown()
		
		
		
	def save_embeddings(self):
		global print_str
		model = self.higashi_model
		model.eval()
		with torch.no_grad():
			ids = torch.arange(1, self.num_list[-1] + 1).long().to(device, non_blocking=True).view(-1)
			embeddings = []
			for j in range(math.ceil(len(ids) / self.batch_size)):
				x = ids[j * self.batch_size:min((j + 1) * self.batch_size, len(ids))]
				
				embed = self.node_embedding_init(x)
				embed = embed.detach().cpu().numpy()
				embeddings.append(embed)
			
			embeddings = np.concatenate(embeddings, axis=0)
			for i in range(len(self.num_list)):
				start = 0 if i == 0 else self.num_list[i - 1]
				static = embeddings[int(start):int(self.num_list[i])]
				
				if i == 0:
					try:
						old_static = np.load(os.path.join(self.embed_dir, "%s_%d_origin.npy" % (self.embedding_name, i)))
						update_rate = np.sum((old_static - static) ** 2, axis=-1) / np.sum(old_static ** 2, axis=-1)
						print_str += "update_rate: %f\t%f\t" % (np.min(update_rate), np.max(update_rate))
					except Exception as e:
						pass
					self.cell_embeddings = static
				np.save(os.path.join(self.embed_dir, "%s_%d_origin.npy" % (self.embedding_name, i)), static)
		
		torch.cuda.empty_cache()
		return embeddings
	
	def get_cell_neighbor_be(self, start=1):
		v = self.cell_embeddings if not self.pre_cell_embed else np.load(self.pre_cell_embed)
		distance = pairwise_distances(v, metric='euclidean')
		distance_sorted = np.sort(distance, axis=-1)
		distance /= np.quantile(distance_sorted[:, 1:self.neighbor_num].reshape((-1)), q=0.25)
		
		cell_neighbor_list_local = [[] for i in range(self.num[0] + 1)]
		cell_neighbor_weight_list_local = [[] for i in range(self.num[0] + 1)]
		if "batch_id" in self.config:
			label = np.array(self.label_info[self.config["batch_id"]])
			batches = np.unique(label)
			equal_num = int(math.ceil((self.neighbor_num - 1) / (len(batches) - 1)))
			
			indexs = [np.where(label == b)[0] for b in batches]
			
			for i, d in enumerate(tqdm(distance)):
				neighbor = []
				weight = []
				b_this = label[i]
				for j in range(len(batches)):
					if b_this == batches[j]:
						neighbor.append(indexs[j][np.argsort(d[indexs[j]])][:1])
						weight.append(d[neighbor[-1]])
					else:
						neighbor.append(indexs[j][np.argsort(d[indexs[j]])][:equal_num])
						weight.append(d[neighbor[-1]])
				
				# print(neighbor, weight)
				
				weight = [w / (np.mean(w) + 1e-15) for w in weight]
				neighbor = np.concatenate(neighbor)
				weight = np.concatenate(weight)
				
				# print(neighbor, weight)
				
				neighbor = neighbor[np.argsort(weight)]
				weight = np.sort(weight)
				
				index = np.random.permutation(len(neighbor) - 1)
				neighbor[1:] = neighbor[1:][index]
				weight[1:] = weight[1:][index]
				neighbor = neighbor[:self.neighbor_num]
				weight = weight[:self.neighbor_num]
				neighbor = neighbor[np.argsort(weight)]
				weight = np.sort(weight)
				
				new_w = np.exp(-weight[start:])
				new_w /= np.sum(new_w)
				cell_neighbor_list_local[i + 1] = (neighbor + 1)[start:]
				cell_neighbor_weight_list_local[i + 1] = (new_w)
		
		return np.array(cell_neighbor_list_local, dtype='object'), np.array(cell_neighbor_weight_list_local, dtype='object')
	
	def get_cell_neighbor(self, start=1):
		v = self.cell_embeddings if not self.pre_cell_embed else np.load(self.pre_cell_embed)
		distance = pairwise_distances(v, metric='euclidean')
		distance_sorted = np.sort(distance, axis=-1)
		distance /= np.quantile(distance_sorted[:, 1:self.neighbor_num].reshape((-1)), q=0.25)
		cell_neighbor_list_local = [[] for i in range(self.num[0] + 1)]
		cell_neighbor_weight_list_local = [[] for i in range(self.num[0] + 1)]
		for i, d in enumerate(tqdm(distance)):
			neighbor = np.argsort(d)[:self.neighbor_num]
			weight = np.sort(d)[:self.neighbor_num]
			
			neighbor_new = neighbor
			new_w = weight
			
			new_w = np.exp(-new_w[start:])
			new_w /= np.sum(new_w)
			
			cell_neighbor_list_local[i + 1] = (neighbor_new + 1)[start:]
			cell_neighbor_weight_list_local[i + 1] = (new_w)
		
		return (np.array(cell_neighbor_list_local, dtype='object'),
				np.array(cell_neighbor_weight_list_local, dtype='object'))
	
	
	def train_for_embeddings(self, max_epochs=None):
		global steps, pair_ratio
		
		optimizer = torch.optim.Adam(
			list(self.higashi_model.parameters()) + list(self.node_embedding_init.parameters()),
			lr=1e-3)
		self.alpha = 1.0
		self.beta = 1e-2
		
		pair_ratio = 0.0
		self.dynamic_pair_ratio = False
		
		steps = 1
		# First round, no cell dependent GNN
		# print ("Pre-training")
		# use_recon = False
		# higashi_model.only_distance=True
		# train(higashi_model,
		#       loss=loss,
		#       training_data_generator=training_data_generator,
		#       validation_data_generator=validation_data_generator,
		#       optimizer=[optimizer], epochs=5,
		#       load_first=False, save_embed=False)
		
		pair_ratio = 0.0
		if mem_efficient_flag:
			self.dynamic_pair_ratio = True
		else:
			self.dynamic_pair_ratio = False
		
		# Training Stage 1
		print("First stage training")
		self.higashi_model.only_distance = False
		self.use_recon = True
		self.train(
		      training_data_generator=self.training_data_generator,
		      validation_data_generator=self.validation_data_generator,
		      optimizer=[optimizer], epochs=self.embedding_epoch if max_epochs is None else max_epochs,
		      load_first=False, save_embed=True, save_name="_stage1")
		
		checkpoint = {
			'model_link': self.higashi_model.state_dict()}
		
		torch.save(checkpoint, self.save_path + "_stage1")
		torch.save(self.higashi_model, self.save_path + "_stage1_model")
	
	
	def train_for_imputation_nbr_0(self):
		self.train_for_imputation_no_nbr()
	
	def train_for_imputation_no_nbr(self):
		global steps, pair_ratio
		# Loading Stage 1
		del self.higashi_model, self.node_embedding_init
		self.higashi_model = torch.load(self.save_path + "_stage1_model", map_location=self.current_device)
		self.node_embedding_init = self.higashi_model.encode1.static_nn
		self.save_embeddings()
		self.node_embedding_init.off_hook([0])
		
		max_distance = self.config['maximum_distance']
		if max_distance < 0:
			max_bin = int(1e5)
		else:
			max_bin = int(max_distance / self.res)
		
		min_distance = self.config['minimum_distance']
		if min_distance < 0:
			min_bin = 0
		else:
			min_bin = int(min_distance / self.res)
		

		self.training_data_generator.filter_edges(min_bin, max_bin)
		self.validation_data_generator.filter_edges(min_bin, max_bin)
		
		self.alpha = 1.0
		self.beta = 1e-3
		self.dynamic_pair_ratio = False
		self.use_recon = False
		self.contractive_flag = False
		self.contractive_loss_weight = 0.0
		
		if mem_efficient_flag:
			pair_ratio = 0.5
		else:
			pair_ratio = 0.0
		
		remove_flag = True
		node_embedding2 = GraphSageEncoder_with_weights(features=self.node_embedding_init,
		                                                linear_features=self.node_embedding_init,
		                                                feature_dim=self.dimensions,
		                                                embed_dim=self.dimensions,
		                                                num_sample=8, gcn=False, num_list=num_list,
		                                                transfer_range=0, start_end_dict=start_end_dict,
		                                                pass_pseudo_id=False, remove=remove_flag,
		                                                pass_remove=False).to(device, non_blocking=True)
		
		self.higashi_model.encode1.dynamic_nn = node_embedding2
		
		optimizer = torch.optim.Adam(list(self.higashi_model.parameters()) + list(self.node_embedding_init.parameters()),
		                             lr=1e-3)
		
		# Second round, with cell dependent GNN, but no neighbors
		steps = 2
		# Training Stage 2
		print("Second stage training")
		self.train(
		      training_data_generator=self.training_data_generator,
		      validation_data_generator=self.validation_data_generator,
		      optimizer=[optimizer], epochs=self.no_nbr_epoch,
		      load_first=False, save_embed=False,
		      save_name="_stage2")
		
		checkpoint = {
			'model_link': self.higashi_model.state_dict()}
		
		torch.save(checkpoint, self.save_path + "_stage2")
		torch.save(self.higashi_model, self.save_path + "_stage2_model")
		
		
	def impute_no_nbr(self):
		# 	# Loading Stage 2
		del self.higashi_model
		self.higashi_model = torch.load(self.save_path + "_stage2_model", map_location=self.current_device)
		self.node_embedding_init = self.higashi_model.encode1.static_nn
		if self.non_para_impute:
			cell_id_all = [np.arange(self.num[0])]
			impute_process(self.config_path, self.higashi_model, "%s_nbr_%d_impute" % (self.embedding_name, 0), self.mode, 0,
			               self.num[0],
			               os.path.join(self.temp_dir, "sparse_nondiag_adj_nbr_1.npy"))
		else:
			impute_pool = ProcessPoolExecutor(max_workers=self.gpu_num)
			torch.save(self.higashi_model, self.save_path + "_stage2_model")
			cell_id_all = np.arange(self.num[0])
			cell_id_all = np.array_split(cell_id_all, self.gpu_num - 1)
			select_gpus = get_free_gpu(self.gpu_num - 1, change_cur=False)
			for i in range(self.gpu_num - 1):
				impute_pool.submit(mp_impute, self.config_path,
				                   self.save_path + "_stage2_model",
				                   "%s_nbr_%d_impute_part_%d" % (self.embedding_name, 0, i),
				                   self.mode, np.min(cell_id_all[i]),
				                   np.max(cell_id_all[i]) + 1,
				                   os.path.join(self.temp_dir, "sparse_nondiag_adj_nbr_1.npy"),
				                   None,
				                   select_gpus[i])

			impute_pool.shutdown(wait=True)
			linkhdf5("%s_nbr_%d_impute" % (self.embedding_name, 0), cell_id_all, self.temp_dir, self.impute_list,
			         None)

	def train_for_imputation_with_nbr(self):
		del self.higashi_model
		global cell_neighbor_list, cell_neighbor_weight_list, steps, weight_dict, pair_ratio, sparse_chrom_list_GCN, weighted_adj
		self.higashi_model = torch.load(self.save_path + "_stage2_model", map_location=self.current_device)
		self.node_embedding_init = self.higashi_model.encode1.static_nn
		self.save_embeddings()
		
		nbr_mode = 0
		max_distance = self.config['maximum_distance']
		if max_distance < 0:
			max_bin = int(1e5)
		else:
			max_bin = int(max_distance / self.res)
		
		min_distance = self.config['minimum_distance']
		if min_distance < 0:
			min_bin = 0
		else:
			min_bin = int(min_distance / self.res)
		
		self.training_data_generator.filter_edges(min_bin, max_bin)
		self.validation_data_generator.filter_edges(min_bin, max_bin)
		
		self.alpha = 1.0
		self.beta = 1e-3
		self.dynamic_pair_ratio = False
		self.use_recon = False
		self.contractive_flag = False
		self.contractive_loss_weight = 0.0
		
		if mem_efficient_flag:
			pair_ratio = 0.5
		else:
			pair_ratio = 0.0
		
		# Training Stage 3
		print("getting cell nbr's nbr list")
		
		if self.remove_be_flag and ("batch_id" in self.config):
			cell_neighbor_list, cell_neighbor_weight_list = self.get_cell_neighbor_be(nbr_mode)
		else:
			cell_neighbor_list, cell_neighbor_weight_list = self.get_cell_neighbor(nbr_mode)
		
		weight_dict = {}
		print(cell_neighbor_list[:10], cell_neighbor_weight_list[:10])
		
		for i in trange(len(cell_neighbor_list)):
			for c, w in zip(cell_neighbor_list[i], cell_neighbor_weight_list[i]):
				weight_dict[(c, i)] = w
		weighted_adj = True
		
		if self.precompute_weighted_nbr:
			new_sparse_chrom_list = [[] for i in range(len(sparse_chrom_list))]
			for c, chrom in enumerate(self.chrom_list):
				new_cell_chrom_list = []
				for cell in np.arange(num_list[0]) + 1:
					mtx = 0
					for nbr_cell in cell_neighbor_list[cell]:
						balance_weight = weight_dict[(nbr_cell, cell)]
						mtx = mtx + balance_weight * sparse_chrom_list[c][nbr_cell - 1]
					mtx = csr_matrix(mtx)
					new_cell_chrom_list.append(mtx)
				new_cell_chrom_list = np.array(new_cell_chrom_list)
				new_sparse_chrom_list[c] = new_cell_chrom_list
			new_sparse_chrom_list = np.array(new_sparse_chrom_list)
			sparse_chrom_list_GCN = new_sparse_chrom_list
		
		np.save(os.path.join(self.temp_dir, "weighted_info.npy"), np.array([cell_neighbor_list, weight_dict]),
		        allow_pickle=True)
		
		
		
		optimizer = torch.optim.Adam(self.higashi_model.parameters(), lr=1e-3)

		steps = 3
		print("Final stage training")
		self.train(training_data_generator=self.training_data_generator,
		      validation_data_generator=self.validation_data_generator,
		      optimizer=[optimizer], epochs=self.with_nbr_epoch,
		      load_first=False, save_name="_stage3", save_embed=False)
		
		checkpoint = {
			'model_link': self.higashi_model.state_dict()}
		
		torch.save(checkpoint, self.save_path + "_stage3")
		torch.save(self.higashi_model, self.save_path + "_stage3_model")
	
	
	def impute_with_nbr(self):
		del self.higashi_model
		# Loading Stage 3
		self.higashi_model = torch.load(self.save_path + "_stage3_model", map_location=self.current_device)
		self.node_embedding_init = self.higashi_model.encode1.static_nn
		# Impute Stage 3
		
		if self.non_para_impute:
			cell_id_all = [np.arange(self.num[0])]
			impute_process(self.config_path, self.higashi_model, "%s_nbr_%d_impute" % (self.embedding_name, self.neighbor_num - 1),
			               self.mode, 0,
			               self.num[0], os.path.join(self.temp_dir, "sparse_nondiag_adj_nbr_1.npy"),
			               os.path.join(self.temp_dir, "weighted_info.npy"))
		else:
			impute_pool = ProcessPoolExecutor(max_workers=self.gpu_num)
			select_gpus = get_free_gpu(self.gpu_num - 1, change_cur=False)
			print("select gpus", select_gpus)
			torch.save(self.higashi_model, self.save_path + "_stage3_model")
			cell_id_all = np.arange(self.num[0])
			cell_id_all = np.array_split(cell_id_all, self.gpu_num - 1)
			for i in range(self.gpu_num - 2, -1, -1):
				impute_pool.submit(mp_impute, self.config_path,
				                   self.save_path + "_stage3_model",
				                   "%s_nbr_%d_impute_part_%d" % (self.embedding_name, self.neighbor_num - 1, i),
				                   self.mode, np.min(cell_id_all[i]),
				                   np.max(cell_id_all[i]) + 1,
				                   os.path.join(self.temp_dir, "sparse_nondiag_adj_nbr_1.npy"),
				                   os.path.join(self.temp_dir, "weighted_info.npy"),
				                   select_gpus[i])
			
			impute_pool.shutdown(wait=True)
			
			# When the 1nb imputation is there and nbr_mode=1 (itself is not included during learning), add the predicted values with only 1nb to the neighbor version.
			linkhdf5("%s_nbr_%d_impute" % (self.embedding_name, self.neighbor_num - 1), cell_id_all, self.temp_dir, self.impute_list,
			         None)
				
	
	def fetch_cell_embeddings(self):
		if self.cell_embeddings is not None:
			return self.cell_embeddings
		else:
			print ("Loading from last training results")
			self.cell_embeddings = np.load(os.path.join(self.embed_dir, "%s_%d_origin.npy" % (self.embedding_name, 0)))
			return self.cell_embeddings
		
	def fetch_map(self, chrom, cell):
		c = self.chrom_list.index(chrom)
		s, e = self.chrom_start_end[c]
		size = e - s
		try:
			with h5py.File(os.path.join(self.temp_dir, "%s_%s_nbr_%d_impute.hdf5" % (chrom, self.embedding_name, 0)), "r") as f:
				coordinates = np.array(f['coordinates']).astype('int')
				p = np.array(f["cell_%d" % cell])

				m1 = csr_matrix((p, (coordinates[:, 0], coordinates[:, 1])), shape=(size, size), dtype='float32')
				m1 = m1 + m1.T
		except Exception as e:
			m1 = np.zeros((size, size))
			print ("No 0 nbr imputation for %s %d" % (chrom, cell))

		try:
			with h5py.File(os.path.join(self.temp_dir, "%s_%s_nbr_%d_impute.hdf5" % (chrom, self.embedding_name, self.neighbor_num - 1)), "r") as f:
				coordinates = np.array(f['coordinates']).astype('int')
				p = np.array(f["cell_%d" % cell])

				m2 = csr_matrix((p, (coordinates[:, 0], coordinates[:, 1])), shape=(size, size), dtype='float32')
				m2 = m2 + m2.T
		except Exception as e:
			m2 = np.zeros((size, size))
			print ("No %d nbr imputation for %s %d" % (self.neighbor_num, chrom, cell))

		if chrom not in self.ori_sparse_list:
			self.ori_sparse_list[chrom] = np.load(os.path.join(self.temp_dir, "raw", "%s_sparse_adj.npy" % chrom), allow_pickle=True)
		
		m3 = self.ori_sparse_list[chrom][cell]
		
		return m3, m1, m2
	
if __name__ == '__main__':
	# Get parameters from config file
	args = parse_args()
	higashi = Higashi(args.config)
	higashi.process_data()
	higashi.prep_model()
	higashi.train_for_embeddings()
	higashi.train_for_imputation_nbr_0()
	higashi.impute_no_nbr()
	higashi.train_for_imputation_with_nbr()
	higashi.impute_with_nbr()
		