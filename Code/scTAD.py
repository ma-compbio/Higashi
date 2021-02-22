from Higashi_backend.utils import *
from Higashi_analysis.Higashi_analysis import *
from Higashi_analysis.Higashi_TAD import *
import h5py
import os

import pickle
import pandas as pd
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
	parser = argparse.ArgumentParser(description="Higashi single cell TAD calling")
	parser.add_argument('-c', '--config', type=str, default="../config_dir/config_Ren_TAD.JSON")
	parser.add_argument('-n', '--neighbor', type=bool, default=True)
	parser.add_argument('--window_ins', type=int, default=1000000)
	parser.add_argument('--window_tad', type=int, default=500000)
	
	return parser.parse_args()


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
	gap_tab.columns = ['chrom', 'start', 'end', 'sth', 'type']
	gap_list = gap_tab[(gap_tab["chrom"] == chrom) & (gap_tab["type"] == "acen")]
	start = np.floor((np.array(gap_list['start']) - 1000000) / res).astype('int')
	end = np.ceil((np.array(gap_list['end']) + 1000000) / res).astype('int')
	
	for s, e in zip(start, end):
		a[s:e, :] = 1
		a[:, s:e] = 1
	a[gap, :] = 1
	a[:, gap] = 1
	
	return a


def kth_diag_indices(a, k):
	rows, cols = np.diag_indices_from(a)
	if k < 0:
		return rows[-k:], cols[:k]
	elif k > 0:
		return rows[:-k], cols[k:]
	else:
		return rows, cols


def gen_tad_and_calibrate(chrom):
	# print("generating single cell scores and boundaries (before calibration)")
	origin_sparse = np.load(os.path.join(temp_dir, "%s_sparse_adj.npy" % chrom), allow_pickle=True)
	size = origin_sparse[0].shape[0]
	mask = create_mask((int(1e5)), chrom, origin_sparse)
	
	bulk1 = np.array(np.sum(origin_sparse, axis=0).todense())
	mask = (np.ones_like(bulk1) - mask)
	bulk1 *= mask
	
	use_rows = np.where(np.sum(bulk1, axis=-1) > 0.1 * np.sum(bulk1) / len(bulk1))[0]
	discard_rows = np.where(np.sum(bulk1, axis=-1) <= 0.1 * np.sum(bulk1) / len(bulk1))[0]
	bulk1 = 0
	
	sc_score = []
	sc_border = []
	sc_border_indice = []
	
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
				temp *= 0.0
				
				proba = np.array(impute_f["cell_%d" % i])
				m1[xs.astype('int'), ys.astype('int')] += proba
				m1 = m1 + m1.T
				m1 = sqrt_norm(m1)
				
				
				if args.neighbor:
					proba = np.array(impute_f2["cell_%d" % i])
					temp[xs.astype('int'), ys.astype('int')] += proba
					temp = temp + temp.T
					temp = sqrt_norm(temp)
					temp = m1 + temp
				else:
					temp = m1
					
				temp *= mask
				
				bulk1 += temp
				
				score = insulation_score(temp, windowsize=args.window_ins, res=res)
				score[discard_rows] = 1.0
				border = call_tads(score, windowsize=args.window_tad, res=res)
				sc_score.append(score)
				sc_border_indice.append(border)
				temp1 = np.zeros_like(score)
				temp1[border] = 1
				sc_border.append(temp1)
	
	sc_score = np.array(sc_score)
	sc_border_indice = np.array(sc_border_indice)
	bulk = sqrt_norm(bulk1)
	# bulk =  new_bulk
	bulk_score = insulation_score(bulk, windowsize=args.window_ins, res=res)
	bulk_score[discard_rows] = 1.0
	bulk_tad_b = call_tads(bulk_score, windowsize=args.window_tad, res=res)
	
	K = int(1.5 * len(bulk_tad_b))
	
	sc_score = np.array(sc_score)
	
	shared_boundaries, sc_assignment, calibrated_sc_boundaries = scTAD_calibrator(K, bulk_score.shape[-1],
	                                                                              chrom).fit_transform(
		np.copy(sc_score),
		np.copy(sc_border_indice),
		bulk_tad_b)
	print("finish %s" % chrom)
	
	calibrated_sc_border = []
	for cb in calibrated_sc_boundaries:
		temp = np.zeros_like(score)
		temp[cb] = 1
		calibrated_sc_border.append(temp)
	
	return chrom, np.array(sc_score), np.array(sc_border), np.array(calibrated_sc_border)


def start_call_tads():
	output_file = h5py.File(os.path.join(temp_dir, "scTAD.hdf5"), "w")
	
	result = {}
	for chrom in chrom_list:
		chrom, sc_score, sc_border, calib_sc_border = gen_tad_and_calibrate(chrom)
		# print (sc_score.shape)
		result[chrom] = [sc_score, sc_border, calib_sc_border]
	
	
	bin_chrom_list = []
	bin_start_list = []
	bin_end_list = []
	signal_list = []
	
	grp = output_file.create_group('insulation')
	bin = grp.create_group('bin')
	
	for chrom in chrom_list:
		vec, border, calib_border = result[chrom]
		length = vec.shape[-1]
		bin_chrom_list += [chrom] * length
		bin_start_list.append((np.arange(length)*res).astype('int'))
		bin_end_list.append(((np.arange(length)+1)*res).astype('int'))
		signal_list.append(vec)
	bin.create_dataset('chrom', data=[l.encode('utf8') for l in bin_chrom_list], dtype = h5py.special_dtype(vlen=str))
	bin.create_dataset('start', data = np.concatenate(bin_start_list))
	bin.create_dataset('end', data=np.concatenate(bin_end_list))
	signal_list = np.concatenate(signal_list, axis=-1)
	for cell in trange(len(signal_list)):
		grp.create_dataset("cell_%d" % cell, data=signal_list[cell])
	
	grp = output_file.create_group('tads')
	bin = grp.create_group('bin')
	signal_list = []
	for chrom in chrom_list:
		_, vec, calib_border = result[chrom]
		signal_list.append(vec)
	bin.create_dataset('chrom', data=[l.encode('utf8') for l in bin_chrom_list], dtype=h5py.special_dtype(vlen=str))
	bin.create_dataset('start', data=np.concatenate(bin_start_list))
	bin.create_dataset('end', data=np.concatenate(bin_end_list))
	signal_list = np.concatenate(signal_list, axis=-1)
	for cell in trange(len(signal_list)):
		grp.create_dataset("cell_%d" % cell, data=signal_list[cell])
		
	grp = output_file.create_group('calib_tads')
	bin = grp.create_group('bin')
	signal_list = []
	for chrom in chrom_list:
		_, _, vec = result[chrom]
		signal_list.append(vec)
	bin.create_dataset('chrom', data=[l.encode('utf8') for l in bin_chrom_list], dtype=h5py.special_dtype(vlen=str))
	bin.create_dataset('start', data=np.concatenate(bin_start_list))
	bin.create_dataset('end', data=np.concatenate(bin_end_list))
	signal_list = np.concatenate(signal_list, axis=-1)
	for cell in trange(len(signal_list)):
		grp.create_dataset("cell_%d" % cell, data=signal_list[cell])
	
	output_file.close()

args = parse_args()
config = get_config(args.config)
res = config['resolution']
temp_dir = config['temp_dir']
cytoband_path = config['cytoband_path']
data_dir = config['data_dir']
neighbor_num = config['neighbor_num']
embedding_name = config['embedding_name']
chrom_list = config['impute_list']
tad_dir = os.path.join(temp_dir, "tad")
if not os.path.exists(tad_dir):
	os.mkdir(tad_dir)
start_call_tads()