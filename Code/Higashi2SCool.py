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
	
	cell_info = {}
	off_set = 0
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
			m1[xs.astype('int'), ys.astype('int')] += proba
			m1 = m1 + m1.T
			m1 *= mask
			temp = m1
			
			info = np.where(temp > 0)
			x,y = info[0], info[1]
			x_mask = y >= x
			x, y = x[x_mask], y[x_mask]
			count = temp[x, y]
			if i not in cell_info:
				cell_info[i] = {}
				cell_info[i]['x'] = []
				cell_info[i]['y'] = []
				cell_info[i]['count'] = []
			cell_info[i]['x'].append(x + off_set)
			cell_info[i]['y'].append(y + off_set)
			cell_info[i]['count'].append(count)
		
		off_set += size
		
		
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
	