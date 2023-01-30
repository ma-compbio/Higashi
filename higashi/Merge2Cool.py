import h5py
import os
import argparse
from Higashi_backend.utils import get_config
from Higashi_analysis.Higashi_analysis import *
import numpy as np
from tqdm import tqdm, trange
import pandas as pd


def parse_args():
	parser = argparse.ArgumentParser(description="Higashi visualization tool")
	parser.add_argument('-c', '--config', type=str, default="")
	parser.add_argument('-o', '--output', default="./output")
	parser.add_argument('-n', '--neighbor', default=False, action='store_true')
	parser.add_argument('-l', '--list', default=None)
	parser.add_argument('-t', '--type', default='selected')
	
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
	start = np.floor((np.array(gap_list['start'])) / res).astype('int')
	end = np.ceil((np.array(gap_list['end'])) / res).astype('int')
	
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
genome_reference_path = config['genome_reference_path']
chrom_list = config['impute_list']
output = args.output
raw_dir = os.path.join(temp_dir, "raw")

list1 = args.list

if list1 is None:
	origin_sparse = np.load(os.path.join(raw_dir, "%s_sparse_adj.npy" % chrom_list[-1]), allow_pickle=True)
	cell_list_group = [np.arange(len(origin_sparse))]
	names = [""]
else:
	file_type = list1.split(".")[-1]
	if file_type == 'npy':
		list1 = np.load(list1)
	else:
		list_file = open(list1, "r")
		list1 = []
		for line in list_file.readlines():
			list1.append(line.strip())
		list1 = np.array(list1)

	list_type = args.type
	if list_type == 'selected':
		cell_list_group = [list1.astype('int')]
		names= [""]
	elif list_type == 'group':
		unique_label = np.unique(list1)
		cell_list_group = [np.where(list1 == u)[0] for u in unique_label]
		names = unique_label

for cell_list, name in zip(cell_list_group, names):
	
	print (cell_list)
	bins_chrom = []
	bins_start = []
	bins_end = []
	
	x = []
	y = []
	count = []
	off_set = 0
	for chrom_index, chrom in enumerate(chrom_list):
		origin_sparse = np.load(os.path.join(raw_dir, "%s_sparse_adj.npy" % chrom), allow_pickle=True)
		size = origin_sparse[0].shape[0]
		mask = create_mask(-1, chrom, origin_sparse)
		mask = 1 - mask
		
		bins_chrom += [chrom] * size
		bins_start.append(np.arange(size) * res)
		bins_end.append(np.arange(size) * res + res)
		
		
		if args.neighbor:
			impute_f = h5py.File(
				os.path.join(temp_dir, "%s_%s_nbr_%d_impute.hdf5" % (chrom, embedding_name, neighbor_num)),
				"r")
		else:
			impute_f = h5py.File(os.path.join(temp_dir, "%s_%s_nbr_0_impute.hdf5" % (chrom, embedding_name)), "r")
		
		coordinates = impute_f['coordinates']
		xs, ys = coordinates[:, 0], coordinates[:, 1]
		
		p_all = 0
		x.append(xs + off_set)
		y.append(ys + off_set)
		
		
		
		for i in cell_list:
			proba = np.array(impute_f["cell_%d" % i])
			p_all += proba
		
		m = np.zeros((size, size))
		m[xs, ys] += p_all
		m = m + m.T
		m *= mask
		p_all = m[xs, ys]
		
		
		count.append(p_all / len(cell_list))
		off_set += size
	
	
	
	
	count = np.concatenate(count, axis=0)
	x = np.concatenate(x, axis=0).astype('int')
	y = np.concatenate(y, axis=0).astype('int')
	
	
	bins_chrom = np.array(bins_chrom)
	bins_start = np.array(np.concatenate(bins_start).astype('int32'))
	bins_end = np.array(np.concatenate(bins_end).astype('int32'))
	
	
	f1 = open("./temp1.txt", "w")
	for i in trange(len(bins_chrom)):
		f1.write("%s\t%d\t%d\n" % (bins_chrom[i], bins_start[i], bins_end[i]))
	f1.close()
	
	f2 = open("./temp2.txt", "w")
	for i in trange(len(count)):
		f2.write("%d\t%d\t%f\n" % (x[i], y[i], count[i]))
	f2.close()
	if output[-5:] == '.cool':
		output = output[:-5]
		
	os.system("cooler load temp1.txt temp2.txt %s --format coo --count-as-float" % (output+"_"+name+".cool"))
	os.remove("./temp1.txt")
	os.remove("./temp2.txt")