import h5py
import os
import argparse
from Higashi_backend.utils import  get_config
from Higashi_analysis.Higashi_analysis import *
import numpy as np
from tqdm import tqdm, trange
import pandas as pd

def parse_args():
	parser = argparse.ArgumentParser(description="Higashi visualization tool")
	parser.add_argument('-c', '--config', type=str, default="")
	parser.add_argument('-o', '--output', default="./output.scool")
	parser.add_argument('-n', '--neighbor', default=False, action='store_true')
	
	
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
	
	gap_tab = pd.read_table(config["cytoband_path"], sep="\t", header=None)
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

args = parse_args()
config = get_config(args.config)
res = config['resolution']
temp_dir = config['temp_dir']
neighbor_num = config['neighbor_num']
embedding_name = config['embedding_name']

chrom_list = config['impute_list']
output = args.output
'''
├── bins
├── chroms
└── cells
 ├── cell_id1
 │   ├── bins
 │   ├── chroms
 │   ├── pixels
 │   └── indexes

 bins
 │   ├── chrom (3088281,) int32
 │   ├── start (3088281,) int32
 │   ├── end (3088281,) int32
'''


with h5py.File(output, "w") as output_f:
	chrom_group = output_f.create_group("chroms")
	chrom_group.create_dataset(name="name", data= np.array(chrom_list).astype('|S64'))
	bin_start = 0
	
	bins_chrom = []
	bins_start = []
	bins_end = []
	
	chrom_offset = [0]
	bin_offset = []
	
	
	cell_info = {}
	off_set = 0
	pixel_off_set = 0
	for chrom_index, chrom in enumerate(chrom_list):
		origin_sparse = np.load(os.path.join(temp_dir, "%s_sparse_adj.npy" % chrom), allow_pickle=True)
		size = origin_sparse[0].shape[0]
		mask = create_mask(-1, chrom, origin_sparse)
		
		bins_chrom += [chrom_index] * size
		bins_start.append(np.arange(size) * res)
		bins_end.append(np.arange(size) * res + res)
		
		
		if args.neighbor:
			impute_f = h5py.File(
				os.path.join(temp_dir, "%s_%s_nbr_%d_impute.hdf5" % (chrom, embedding_name, neighbor_num)),
				"r")
		else:
			impute_f = h5py.File(os.path.join(temp_dir, "%s_%s_nbr_1_impute.hdf5" % (chrom, embedding_name)), "r")
				
		coordinates = impute_f['coordinates']
		xs, ys = coordinates[:, 0], coordinates[:, 1]
		cell_list = trange(len(list(impute_f.keys())) - 1)
		for i in cell_list:
			m1 = np.zeros((size, size))
			proba = np.array(impute_f["cell_%d" % i])
			
			if i not in cell_info:
				cell_info[i] = {}
				cell_info[i]['x'] = []
				cell_info[i]['y'] = []
				cell_info[i]['count'] = []
			cell_info[i]['x'].append(xs + off_set)
			cell_info[i]['y'].append(ys + off_set)
			cell_info[i]['count'].append(proba)
		
		_, indice = np.unique(xs, return_index=True)
		bin_offset.append(indice + pixel_off_set)
		pixel_off_set += len(proba)
		off_set += size
		
		chrom_offset.append(off_set)
	
	bin_offset.append(np.array([len(np.concatenate(cell_info[i]['count'], axis=0))]))
	chrom_offset = np.array(chrom_offset)
	bin_offset = np.concatenate(bin_offset)
		
	bins_group = output_f.create_group("bins")
	bins_group.create_dataset(name="chrom", data=np.array(bins_chrom).astype('int32'))
	bins_group.create_dataset(name="start", data=np.concatenate(bins_start).astype('int32'))
	bins_group.create_dataset(name="end", data=np.concatenate(bins_end).astype('int32'))
	
	for cell in cell_list:
		cell_group = output_f.create_group("cell_%d" % cell)
		cell_group['chroms'] = chrom_group
		cell_group['bins'] = bins_group
		pixel_group = cell_group.create_group("pixels")
		pixel_group.create_dataset(name='bin1_id', data = np.concatenate(cell_info[cell]['x'], axis=0).astype('int'))
		pixel_group.create_dataset(name='bin2_id', data=np.concatenate(cell_info[cell]['y'], axis=0).astype('int'))
		pixel_group.create_dataset(name='count', data=np.concatenate(cell_info[cell]['count'], axis=0).astype('float'))
		
		indexes_group = cell_group.create_group("indexes")
		indexes_group.create_dataset(name='chrom_offset', data=chrom_offset.astype('int64'))
		indexes_group.create_dataset(name='bin1_offset', data=bin_offset.astype('int64'))