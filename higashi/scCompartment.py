from Higashi_backend.utils import *
from Higashi_analysis.Higashi_analysis import *
import h5py
from sklearn.preprocessing import MinMaxScaler, quantile_transform
import os

os.environ["OMP_NUM_THREADS"] = "10"
import pandas as pd
import argparse


def parse_args():
	parser = argparse.ArgumentParser(description="Higashi single cell compartment calling")
	parser.add_argument('-c', '--config', type=str, default="../config_dir/config_Ren_221.JSON")
	parser.add_argument('-n', '--neighbor', default=False, action='store_true')
	parser.add_argument('-o', '--output', type=str, default="scCompartment")
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
	if cytoband_path is not None:
		gap_tab = pd.read_table(cytoband_path, sep="\t", header=None)
		gap_tab.columns = ['chrom', 'start', 'end', 'name', 'type']
		
		name = np.array(gap_tab['name'])
		# print (name)
		pqarm = np.array([str(s)[0] for s in name])
		gap_tab['pq_arm'] = pqarm
		gap_tab['length'] = gap_tab['end'] - gap_tab['start']
		summarize = gap_tab.groupby(['chrom', 'pq_arm']).sum().reset_index()
		# print (summarize)
		
		if np.sum(summarize['pq_arm'] == 'p') > 0:
			split_point = \
			np.ceil(np.array(summarize[(summarize['chrom'] == chrom) & (summarize['pq_arm'] == 'p')]['length']) / res)[0]
		else:
			split_point = -1
		
		gap_list = gap_tab[(gap_tab["chrom"] == chrom) & (gap_tab["type"] == "acen")]
		start = np.floor((np.array(gap_list['start'])) / res).astype('int')
		end = np.ceil((np.array(gap_list['end'])) / res).astype('int')
		
		for s, e in zip(start, end):
			a[s:e, :] = 1
			a[:, s:e] = 1
	else:
		split_point = -1
	a[gap, :] = 1
	a[:, gap] = 1
	
	return a, int(split_point)


def process_one_chrom(chrom):
	# Get the raw sparse mtx list
	origin_sparse = np.load(os.path.join(raw_dir, "%s_sparse_adj.npy" % chrom), allow_pickle=True)
	size = origin_sparse[0].shape[0]
	# find centromere & gaps...
	mask, split_point = create_mask((int(1e5)), chrom, origin_sparse)
	
	bulk1 = np.array(np.sum(origin_sparse, axis=0).todense())
	print(bulk1.shape)
	mask = (np.ones_like(bulk1) - mask)
	bulk1 *= mask
	
	if "bulk_path" in config:
		import cooler
		c = cooler.Cooler("%s::resolutions/%d" % (config['bulk_path'], config['resolution']))
		bulk2 = np.array(c.matrix(sparse=False, balance=False).fetch(chrom)).astype('float')
		bulk2 *= mask
	else:
		bulk2 = None
	
	use_rows_all = []
	
	if split_point >= 20 * 1000000 / res:
		slice_start_list, slice_end_list = [0, split_point], [split_point, len(bulk1)]
	else:
		slice_start_list, slice_end_list = [0], [len(bulk1)]
	
	bulk_compartment_all = []
	real_bulk_compartment_all = []
	
	bulk_model_list = []
	bulk_reverse_list = []
	bulk_slice_list = []
	use_rows_list = []
	temp_compartment_list_zscore = []
	temp_compartment_list_quantile = []
	
	for slice_start, slice_end in zip(slice_start_list, slice_end_list):
		
		bulk1_slice = bulk1[slice_start:slice_end, :]
		bulk1_slice = bulk1_slice[:, slice_start:slice_end]
		use_rows = np.where(np.sum(bulk1_slice > 0, axis=-1) > 0.01 * len(bulk1_slice))[0]
		if len(use_rows) <= 1:
			print("no reliable bins in slice:", slice_start, slice_end)
			continue
		use_rows_all.append(np.arange(slice_start, slice_end)[use_rows])
		use_rows_list.append(use_rows)
		bulk1_slice = bulk1_slice[use_rows, :]
		bulk1_slice = bulk1_slice[:, use_rows]
		
		bulk_slice_list.append(bulk1_slice)
		bulk_expect = []
		for k in range(len(bulk1_slice)):
			diag = np.diag(bulk1_slice, k)
			bulk_expect.append(np.mean(diag))
		
		
		bulk_compartment, model = compartment(bulk1_slice, return_PCA=True)
		
		if bulk2 is not None:
			bulk2_slice = bulk2[slice_start:slice_end, :]
			bulk2_slice = bulk2_slice[:, slice_start:slice_end]
			bulk2_slice = bulk2_slice[use_rows, :]
			bulk2_slice = bulk2_slice[:, use_rows]
			real_bulk_compartment, model = compartment(bulk2_slice, return_PCA=True)
		else:
			real_bulk_compartment = None
			
		reverse_flag = False
		if args.calib:
			calib = np.load(os.path.join(temp_dir, "calib.npy"), allow_pickle=True).item()[chrom].reshape((-1, 1))[slice_start:slice_end][
				use_rows]
			print ("average cpg", np.nanmean(calib[bulk_compartment > np.quantile(bulk_compartment,0.9)]), np.nanmean(calib[bulk_compartment < np.quantile(bulk_compartment,0.1)]))
			if np.nanmean(calib[bulk_compartment > np.quantile(bulk_compartment,0.9)]) < np.nanmean(calib[bulk_compartment < np.quantile(bulk_compartment,0.1)]):
				reverse_flag = True
			if reverse_flag:
				bulk_compartment *= -1
			
			if real_bulk_compartment is not None:
				reverse_flag = False
				if np.nanmean(calib[real_bulk_compartment > np.quantile(real_bulk_compartment, 0.9)]) < np.nanmean(
					calib[real_bulk_compartment < np.quantile(real_bulk_compartment, 0.1)]):
					real_bulk_compartment *= -1
					reverse_flag = True
		bulk_compartment_all.append(bulk_compartment)
		bulk_reverse_list.append(reverse_flag)
		bulk_model_list.append(model)
		real_bulk_compartment_all.append(real_bulk_compartment)
		
		
	
		
	if args.neighbor:
		impute_f = h5py.File(os.path.join(temp_dir, "%s_%s_nbr_%d_impute.hdf5" % (chrom, embedding_name, neighbor_num)),
		               "r")
	else:
		impute_f =  h5py.File(os.path.join(temp_dir, "%s_%s_nbr_0_impute.hdf5" % (chrom, embedding_name)),
		               "r")
	temp_compartment_list_all = [[] for i in range(len(use_rows_list))]
	coordinates = impute_f['coordinates']
	xs, ys = coordinates[:, 0], coordinates[:, 1]
	cell_list = trange(len(list(impute_f.keys())) - 1)
	temp = np.zeros((size, size))
	
	for i in cell_list:
		temp *= 0.0
		proba = np.array(impute_f["cell_%d" % i])
		temp[xs.astype('int'), ys.astype('int')] += proba
		temp = temp + temp.T
		temp *= mask
		
		for j in range(len(use_rows_list)):
			slice_start, slice_end = slice_start_list[j], slice_end_list[j]
			temp_slice = temp[slice_start:slice_end, :]
			temp_slice = temp_slice[:, slice_start:slice_end]
			temp_select = temp_slice[use_rows_list[j], :]
			temp_select = temp_select[:, use_rows_list[j]]
			# temp_select = rankmatch(temp_select, bulk_slice_list[j])
			temp_compartment = compartment(temp_select, False, bulk_model_list[j], None)
			if bulk_reverse_list[j]:
				temp_compartment = -1 * temp_compartment
			temp_compartment_list_all[j].append(temp_compartment.reshape((-1)))
	for j in range(len(use_rows_list)):
		temp_compartment_list_all[j] = np.stack(temp_compartment_list_all[j], axis=0)
		temp_compartment_list_quantile.append(quantile_transform(temp_compartment_list_all[j], output_distribution='uniform',
		                                           n_quantiles=int(temp_compartment_list_all[j].shape[-1] * 1.0), axis=1))
		
		temp_compartment_list_zscore.append(zscore(temp_compartment_list_all[j], axis=1))

	# print(bulk_compartment.shape, temp_compartment_list.shape)
	
	
	real_bulk_compartment = np.concatenate(real_bulk_compartment_all, axis=0) if bulk2 is not None else None
	bulk_compartment = np.concatenate(bulk_compartment_all, axis=0)
	temp_compartment_list = np.concatenate(temp_compartment_list_all, axis=-1)
	temp_compartment_list_zscore = np.concatenate(temp_compartment_list_zscore, axis=-1)
	temp_compartment_list_quantile = np.concatenate(temp_compartment_list_quantile, axis=-1)
	use_rows = np.concatenate(use_rows_all, axis=0)
	print (chrom, "finished")
	return real_bulk_compartment, bulk_compartment, temp_compartment_list, temp_compartment_list_zscore, temp_compartment_list_quantile, chrom, use_rows, size


def process_calib_file(file_path):
	tab = pd.read_table(file_path , sep="\t", header=None)
	tab.columns = ['chrom', 'bin', 'value']
	# print(tab)
	chrom_start_end = np.load(os.path.join(temp_dir, "chrom_start_end.npy"))
	tab['chrom'] = np.array(tab['chrom']).astype('str')
	calib_result = {}
	for i, chrom in enumerate(chrom_list):
		temp = tab[tab['chrom'] == chrom]
		size = chrom_start_end[i, 1] - chrom_start_end[i, 0]
		vec = np.zeros(size)
		indice = (np.array(temp['bin'] / res)).astype('int')
		v = np.array(temp['value'])
		v[v == -1] = np.nan
		vec[indice] = v
		calib_result[chrom] = vec
	np.save(os.path.join(temp_dir, "calib.npy"), calib_result, allow_pickle=True)


def start_call_compartment():
	p_list = []
	pool = ProcessPoolExecutor(max_workers=10)
	output = args.output
	if ".hdf5" not in output:
		output += ".hdf5"
	with h5py.File(os.path.join(temp_dir, output), "w") as output_f:
		result = {}
		for chrom in chrom_list:
			real_bulk_compartment, bulk_compartment, temp_compartment_list, temp_compartment_zscore, temp_compartment_quantile, chrom, use_rows, size = process_one_chrom(chrom)
		# 	p_list.append(pool.submit(process_one_chrom, chrom))
		# 
		# 
		# for p in as_completed(p_list):
		# 	real_bulk_compartment, bulk_compartment, temp_compartment_list, temp_compartment_zscore, temp_compartment_quantile, chrom, use_rows, size = p.result()
			result[chrom] = [real_bulk_compartment, bulk_compartment, temp_compartment_list, temp_compartment_zscore, temp_compartment_quantile, use_rows, size]
			
		bin_chrom_list = []
		bin_start_list = []
		bin_end_list = []
		bulk_cp_all = []
		real_bulk_cp_all = []
		sc_cp_all = []
		sc_cp_raw = []
		sc_cp_zscore = []
		grp = output_f.create_group('compartment')
		bin = grp.create_group('bin')
		
		for chrom in chrom_list:
			real_bulk_compartment, bulk_compartment, temp_compartment_list, temp_compartment_zscore, temp_compartment_quantile, use_rows, size = result[chrom]
			# print (use_rows)
			length = size
			bin_chrom_list += [chrom] * len(use_rows)
			bin_start_list.append((np.arange(length) * res).astype('int')[use_rows])
			bin_end_list.append(((np.arange(length) + 1) * res).astype('int')[use_rows])
			bulk_cp_all.append(bulk_compartment)
			real_bulk_cp_all.append(real_bulk_compartment)
			sc_cp_all.append(temp_compartment_quantile)
			sc_cp_raw.append(temp_compartment_list)
			sc_cp_zscore.append(temp_compartment_zscore)
			
		bin.create_dataset('chrom', data=[l.encode('utf8') for l in bin_chrom_list],
		                   dtype=h5py.special_dtype(vlen=str))
		bin.create_dataset('start', data=np.concatenate(bin_start_list))
		bin.create_dataset('end', data=np.concatenate(bin_end_list))
		
		bulk_cp_all = np.concatenate(bulk_cp_all, axis=0)
		grp.create_dataset("bulk", data=bulk_cp_all)
		
		if real_bulk_compartment is not None:
			real_bulk_cp_all = np.concatenate(real_bulk_cp_all, axis=0)
			grp.create_dataset("real_bulk", data=real_bulk_cp_all)
		
		sc_cp_all = np.concatenate(sc_cp_all, axis=-1)
		sc_cp_raw = np.concatenate(sc_cp_raw, axis=-1)
		sc_cp_zscore = np.concatenate(sc_cp_zscore, axis=-1)
		
		for cell in range(len(sc_cp_all)):
			grp.create_dataset("cell_%d" % cell, data=sc_cp_all[cell])
		
		grp = output_f.create_group('compartment_raw')
		bin = grp.create_group('bin')
		bin.create_dataset('chrom', data=[l.encode('utf8') for l in bin_chrom_list],
		                   dtype=h5py.special_dtype(vlen=str))
		bin.create_dataset('start', data=np.concatenate(bin_start_list))
		bin.create_dataset('end', data=np.concatenate(bin_end_list))
		for cell in range(len(sc_cp_all)):
			grp.create_dataset("cell_%d" % cell, data=sc_cp_raw[cell])
		
		grp = output_f.create_group('compartment_zscore')
		bin = grp.create_group('bin')
		bin.create_dataset('chrom', data=[l.encode('utf8') for l in bin_chrom_list],
		                   dtype=h5py.special_dtype(vlen=str))
		bin.create_dataset('start', data=np.concatenate(bin_start_list))
		bin.create_dataset('end', data=np.concatenate(bin_end_list))
		for cell in range(len(sc_cp_all)):
			grp.create_dataset("cell_%d" % cell, data=sc_cp_zscore[cell])
	output_f.close()
	pool.shutdown(wait=True)




args = parse_args()
print (args)
config = get_config(args.config)
res = config['resolution']
data_dir = config['data_dir']
temp_dir = config['temp_dir']
raw_dir = os.path.join(temp_dir, "raw")
if 'cytoband_path' in config:
	cytoband_path = config['cytoband_path']
else:
	cytoband_path = None
neighbor_num = config['neighbor_num']
embedding_name = config['embedding_name']

chrom_list = config['impute_list']
chrom_list = np.array(chrom_list)


if args.calib:
	process_calib_file(args.calib_file)
start_call_compartment()