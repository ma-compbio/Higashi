from Higashi_backend.utils import *
from Higashi_analysis.Higashi_analysis import *
import h5py
from sklearn.preprocessing import MinMaxScaler, quantile_transform
import os

os.environ["OMP_NUM_THREADS"] = "10"

import pickle
import pandas as pd
import argparse


def parse_args():
	parser = argparse.ArgumentParser(description="Higashi single cell compartment calling")
	parser.add_argument('-c', '--config', type=str, default="../config_dir/config_Ren_221.JSON")
	parser.add_argument('-n', '--neighbor', type=bool, default=True)
	parser.add_argument('--calib_file', type=str, default="./calib.bed")
	parser.add_argument('--calib', action='store_true')
	return parser.parse_args()


def rankmatch(from_mtx, to_mtx):
	temp = np.sort(to_mtx.reshape((-1)))
	temp2 = from_mtx.reshape((-1))
	order = np.argsort(temp2)
	temp2[order] = temp
	return temp2.reshape((len(from_mtx), -1))


def create_mask(k=30, chrom="chr1", origin_sparse=None):
	final = np.array(np.sum(origin_sparse, axis=0).todense())
	size = origin_sparse[0].shape[-1]
	a = np.zeros((size, size))
	if k > 0:
		for i in range(min(k, len(a))):
			for j in range(len(a) - i):
				a[j, j + i] = 1
				a[j + i, j] = 1
		a = np.ones_like((a)) - a
	
	gap = np.sum(final, axis=-1, keepdims=False) == 0
	
	gap_tab = pd.read_table(cytoband_path, sep="\t", header=None)
	gap_tab.columns = ['chrom', 'start', 'end', 'name', 'type']
	
	name = np.array(gap_tab['name'])
	pqarm = np.array([s[0] for s in name])
	gap_tab['pq_arm'] = pqarm
	gap_tab['length'] = gap_tab['end'] - gap_tab['start']
	summarize = gap_tab.groupby(['chrom', 'pq_arm']).sum().reset_index()
	# print (summarize)
	
	split_point = \
	np.ceil(np.array(summarize[(summarize['chrom'] == chrom) & (summarize['pq_arm'] == 'p')]['length']) / res)[0]
	
	gap_list = gap_tab[(gap_tab["chrom"] == chrom) & (gap_tab["type"] == "acen")]
	start = np.floor((np.array(gap_list['start'])) / res).astype('int')
	end = np.ceil((np.array(gap_list['end'])) / res).astype('int')
	
	for s, e in zip(start, end):
		a[s:e, :] = 1
		a[:, s:e] = 1
	a[gap, :] = 1
	a[:, gap] = 1
	
	return a, int(split_point)


def process_one_chrom(chrom):
	origin_sparse = np.load(os.path.join(temp_dir, "%s_sparse_adj.npy" % chrom), allow_pickle=True)
	size = origin_sparse[0].shape[0]
	mask, split_point = create_mask((int(1e5)), chrom, origin_sparse)
	
	bulk1 = np.array(np.sum(origin_sparse, axis=0).todense())
	print(bulk1.shape)
	mask = (np.ones_like(bulk1) - mask)
	bulk1 *= mask
	
	bulk_compartment_all = []
	temp_compartment_list_all = []
	use_rows_all = []
	
	if split_point >= 20:
		slice_start_list, slice_end_list = [0, split_point], [split_point, len(bulk1)]
	else:
		slice_start_list, slice_end_list = [0], [len(bulk1)]
	for slice_start, slice_end in zip(slice_start_list, slice_end_list):
		
		bulk1_slice = bulk1[slice_start:slice_end, :]
		bulk1_slice = bulk1_slice[:, slice_start:slice_end]
		use_rows = np.where(np.sum(bulk1_slice > 0, axis=-1) > 0.1 * len(bulk1_slice))[0]
		use_rows_all.append(np.arange(slice_start, slice_end)[use_rows])
		if len(use_rows) <= 1:
			# print("no use", slice_start, slice_end)
			continue
		
		
		# print(bulk1_slice.shape)
		bulk1_slice = bulk1_slice[use_rows, :]
		bulk1_slice = bulk1_slice[:, use_rows]
		# print(bulk1_slice.shape)
		bulk_expect = []
		for k in range(len(bulk1_slice)):
			diag = np.diag(bulk1_slice, k)
			bulk_expect.append(np.mean(diag))
		
		
		bulk_compartment, model = compartment(bulk1_slice, return_PCA=True)
		
		reverse_flag = False
		if args.calib:
			calib = np.load(os.path.join(temp_dir, "%s_calib.npy" % chrom)).reshape((-1, 1))[slice_start:slice_end][
				use_rows]
			if np.mean(calib[bulk_compartment > 0]) < np.mean(calib[bulk_compartment < 0]):
				reverse_flag = True
		
		temp_compartment_list = []
		
		
		with h5py.File(os.path.join(temp_dir, "%s_%s_nbr_1_impute.hdf5" % (chrom, embedding_name)), "r") as impute_f:
			with h5py.File(os.path.join(temp_dir, "%s_%s_nbr_%d_impute.hdf5" % (chrom, embedding_name, neighbor_num)),
			               "r") as impute_f2:
				coordinates = impute_f['coordinates']
				xs, ys = coordinates[:, 0], coordinates[:, 1]
				cell_list = trange(len(list(impute_f.keys())) - 1)
				m1 = np.zeros((size, size))
				temp = np.zeros((size, size))
				
				for i in cell_list:
					m1 *= 0.0
					proba = np.array(impute_f["cell_%d" % i])
					m1[xs.astype('int'), ys.astype('int')] += proba
					m1 = m1 + m1.T
					m1 *= mask
					
					m1_slice = m1[slice_start:slice_end, :]
					m1_slice = m1_slice[:, slice_start:slice_end]
					
					m1_select = m1_slice[use_rows, :]
					m1_select = m1_select[:, use_rows]
					
					if args.neighbor:
						temp *= 0.0
						proba = np.array(impute_f2["cell_%d" % i])
						temp[xs.astype('int'), ys.astype('int')] += proba
						temp = temp + temp.T
						temp *= mask
						
						temp_slice = temp[slice_start:slice_end, :]
						temp_slice = temp_slice[:, slice_start:slice_end]
						
						temp_select = temp_slice[use_rows, :]
						temp_select = temp_select[:, use_rows]
					
						temp_select = m1_select / np.mean(m1_select) + temp_select / np.mean(m1_select)
					else:
						temp_select = m1_select
						
					temp_select = rankmatch(temp_select, bulk1_slice)
					temp_compartment = compartment(temp_select, False, model, None)
					if reverse_flag:
						temp_compartment = -1 * temp_compartment
					temp_compartment_list.append(temp_compartment.reshape((-1)))
		temp_compartment_list = np.stack(temp_compartment_list, axis=0)
		temp_compartment_list = quantile_transform(temp_compartment_list, output_distribution='uniform',
		                                           n_quantiles=int(temp_compartment_list.shape[-1] * 1.0), axis=1)
		bulk_compartment_all.append(bulk_compartment)
		temp_compartment_list_all.append(temp_compartment_list)
		# print(bulk_compartment.shape, temp_compartment_list.shape)
	bulk_compartment = np.concatenate(bulk_compartment_all, axis=0)
	temp_compartment_list = np.concatenate(temp_compartment_list_all, axis=-1)
	use_rows = np.concatenate(use_rows_all, axis=0)
	return bulk_compartment, temp_compartment_list, chrom, use_rows, size


def process_calib_file(file_path):
	tab = pd.read_table(file_path , sep="\t", header=None)
	tab.columns = ['chrom', 'bin', 'value']
	print(tab)
	chrom_start_end = np.load(os.path.join(temp_dir, "chrom_start_end.npy"))
	tab['chrom'] = np.array(tab['chrom']).astype('str')
	for i, chrom in enumerate(chrom_list):
		temp = tab[tab['chrom'] == chrom]
		size = chrom_start_end[i, 1] - chrom_start_end[i, 0]
		vec = np.zeros(size)
		indice = (np.array(temp['bin'] / res)).astype('int')
		v = np.array(temp['value'])
		print(indice, v)
		vec[indice] = v
		print(vec.shape)
		np.save(os.path.join(temp_dir, "%s_calib.npy" % chrom), vec)



def start_call_compartment():
	p_list = []
	pool = ProcessPoolExecutor(max_workers=25)
	with h5py.File(os.path.join(temp_dir, "sc_compartment.hdf5"), "w") as output_f:
		for chrom in chrom_list:
			p_list.append(pool.submit(process_one_chrom, chrom))
		
		result = {}
		
		for p in as_completed(p_list):
			bulk_compartment, temp_compartment_list, chrom, use_rows, size = p.result()
			result[chrom] = [bulk_compartment, temp_compartment_list, use_rows, size]
			
		bin_chrom_list = []
		bin_start_list = []
		bin_end_list = []
		bulk_cp_all = []
		sc_cp_all = []
		grp = output_f.create_group('compartment')
		bin = grp.create_group('bin')
		
		for chrom in chrom_list:
			bulk_compartment, temp_compartment_list, use_rows, size = result[chrom]
			# print (use_rows)
			length = size
			bin_chrom_list += [chrom] * len(use_rows)
			bin_start_list.append((np.arange(length) * res).astype('int')[use_rows])
			bin_end_list.append(((np.arange(length) + 1) * res).astype('int')[use_rows])
			bulk_cp_all.append(bulk_compartment)
			sc_cp_all.append(temp_compartment_list)
			
		bin.create_dataset('chrom', data=[l.encode('utf8') for l in bin_chrom_list],
		                   dtype=h5py.special_dtype(vlen=str))
		bin.create_dataset('start', data=np.concatenate(bin_start_list))
		bin.create_dataset('end', data=np.concatenate(bin_end_list))
		
		bulk_cp_all = np.concatenate(bulk_cp_all, axis=0)
		grp.create_dataset("bulk", data=bulk_cp_all)
		
		sc_cp_all = np.concatenate(sc_cp_all, axis=-1)
		for cell in trange(len(sc_cp_all)):
			grp.create_dataset("cell_%d" % cell, data=sc_cp_all[cell])
	output_f.close()
	pool.shutdown(wait=True)




args = parse_args()
config = get_config(args.config)
res = config['resolution']
data_dir = config['data_dir']
temp_dir = config['temp_dir']
cytoband_path = config['cytoband_path']
neighbor_num = config['neighbor_num']
embedding_name = config['embedding_name']

chrom_list = config['impute_list']
chrom_list = np.array(chrom_list)


if args.calib:
	process_calib_file(args.calib_file)
start_call_compartment()
