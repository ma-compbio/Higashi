from Higashi_backend.utils import *
from Higashi_analysis.Higashi_analysis import *
from Higashi_analysis.Higashi_TAD import *
import h5py
import os
import pandas as pd
import argparse



def parse_args():
	parser = argparse.ArgumentParser(description="Higashi single cell TAD calling")
	parser.add_argument('-c', '--config', type=str, default="../config_dir/config_ercker_10Kb.JSON")
	parser.add_argument('-n', '--neighbor', default=False, action='store_true')
	parser.add_argument('-o', '--output', type=str, default="scTAD")
	parser.add_argument('--window_ins', type=int, default=500000)
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
	if cytoband_path is not None:
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
	
	return a, final


def kth_diag_indices(a, k):
	rows, cols = np.diag_indices_from(a)
	if k < 0:
		return rows[-k:], cols[:k]
	elif k > 0:
		return rows[:-k], cols[k:]
	else:
		return rows, cols


def gen_tad(chrom):
	print("generating single cell scores and boundaries (before calibration) for", chrom)
	origin_sparse = np.load(os.path.join(raw_dir, "%s_sparse_adj.npy" % chrom), allow_pickle=True)
	size = origin_sparse[0].shape[0]
	mask, bulk1 = create_mask((int(1e5)), chrom, origin_sparse)
	
	# bulk1 = np.array(np.sum(origin_sparse, axis=0).todense())
	mask = (np.ones_like(bulk1) - mask)
	bulk1 *= mask
	#
	# use_rows = np.where(np.sum(bulk1, axis=-1) > 0.1 * np.sum(bulk1) / len(bulk1))[0]
	discard_rows = np.where(np.sum(bulk1, axis=-1) <= 0.01 * np.sum(bulk1) / len(bulk1))[0]
	bulk1 = 0
	
	
	sc_score = []
	sc_border = []
	sc_border_indice = []
	
	if args.neighbor:
		impute_f = h5py.File(os.path.join(temp_dir, "%s_%s_nbr_%d_impute.hdf5" % (chrom, embedding_name, neighbor_num)),
		               "r")
	else:
		impute_f = h5py.File(os.path.join(temp_dir, "%s_%s_nbr_0_impute.hdf5" % (chrom, embedding_name)), "r")
		
	coordinates = impute_f['coordinates']
	xs, ys = coordinates[:, 0], coordinates[:, 1]
	
	cell_list = trange(len(list(impute_f.keys())) - 1)
	m1 = np.zeros((size, size))
	for i in cell_list:
		m1 *= 0.0
		
		proba = np.array(impute_f["cell_%d" % i])
		m1[xs.astype('int'), ys.astype('int')] += proba
		m1 = m1 + m1.T
		temp = m1
		temp *= mask
		temp = sqrt_norm(temp)
		
		
		bulk1 += proba
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
	bulk = np.array(csr_matrix((bulk1, (xs, ys)), shape=(size, size)).todense())
	bulk *= mask
	bulk = sqrt_norm(bulk)
	bulk_score = insulation_score(bulk, windowsize=args.window_ins, res=res)
	bulk_score[discard_rows] = 1.0
	bulk_tad_b = call_tads(bulk_score, windowsize=args.window_tad, res=res)
	sc_score = np.array(sc_score)
	
	return chrom, np.array(sc_score), np.array(sc_border),np.array(sc_border_indice), bulk_score, bulk_tad_b
	
	
def calibrate_tad(chrom, sc_score, sc_border, sc_border_indice, bulk_score, bulk_tad_b):
	K = int(1.5 * len(bulk_tad_b))
	shared_boundaries, sc_assignment, calibrated_sc_boundaries = scTAD_calibrator(K, bulk_score.shape[-1],
	                                                                              chrom).fit_transform(
		np.copy(sc_score),
		np.copy(sc_border_indice),
		bulk_tad_b)
	print("finish %s" % chrom)

	calibrated_sc_border = []
	for cb in calibrated_sc_boundaries:
		temp = np.zeros_like(sc_score[0])
		temp[cb] = 1
		calibrated_sc_border.append(temp)
	
	return chrom, np.array(sc_score), np.array(sc_border), np.array(calibrated_sc_border), bulk_score


def start_call_tads():
	p_list = []
	calib_p_list = []
	pool = ProcessPoolExecutor(max_workers=3)
	calib_pool = ProcessPoolExecutor(max_workers=23)
	output = args.output
	if ".hdf5" not in output:
		output += ".hdf5"
	output_file = h5py.File(os.path.join(temp_dir, output), "w")
	
	result = {}
	for chrom in chrom_list:
		p_list.append(pool.submit(gen_tad, chrom))
	for p in as_completed(p_list):
		chrom, sc_score, sc_border,sc_border_indice, bulk_score, bulk_tad_b = p.result()
		calib_sc_border = []
		calib_p_list.append(calib_pool.submit(calibrate_tad, chrom, sc_score, sc_border,sc_border_indice, bulk_score, bulk_tad_b))
	pool.shutdown(wait=True)

	for p in as_completed(calib_p_list):
		chrom, sc_score, sc_border, calib_sc_border, bulk_score = p.result()
		result[chrom] = [sc_score, sc_border, calib_sc_border, bulk_score]
	
	bin_chrom_list = []
	bin_start_list = []
	bin_end_list = []
	signal_list = []
	bulk_list = []
	grp = output_file.create_group('insulation')
	bin = grp.create_group('bin')
	
	for chrom in chrom_list:
		vec, border, calib_border, bulk = result[chrom]
		length = vec.shape[-1]
		bin_chrom_list += [chrom] * length
		bin_start_list.append((np.arange(length)*res).astype('int'))
		bin_end_list.append(((np.arange(length)+1)*res).astype('int'))
		signal_list.append(vec)
		bulk_list.append(bulk)
	bin.create_dataset('chrom', data=[l.encode('utf8') for l in bin_chrom_list], dtype = h5py.special_dtype(vlen=str))
	bin.create_dataset('start', data = np.concatenate(bin_start_list))
	bin.create_dataset('end', data=np.concatenate(bin_end_list))
	signal_list = np.concatenate(signal_list, axis=-1)
	bulk_list = np.concatenate(bulk_list, axis=-1)
	for cell in trange(len(signal_list)):
		grp.create_dataset("cell_%d" % cell, data=signal_list[cell])
	grp.create_dataset("bulk", data=bulk_list)
	grp = output_file.create_group('tads')
	bin = grp.create_group('bin')
	signal_list = []
	for chrom in chrom_list:
		_, vec, calib_border, _ = result[chrom]
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
		_, _, vec, _ = result[chrom]
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
raw_dir = os.path.join(temp_dir, "raw")
if 'cytoband_path' in config:
	cytoband_path = config['cytoband_path']
else:
	cytoband_path = None
data_dir = config['data_dir']
neighbor_num = config['neighbor_num']
embedding_name = config['embedding_name']
chrom_list = config['impute_list']
tad_dir = os.path.join(temp_dir, "tad")
if not os.path.exists(tad_dir):
	os.mkdir(tad_dir)
start_call_tads()