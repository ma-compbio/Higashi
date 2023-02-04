import h5py
import os
import argparse
from Higashi_backend.utils import get_config
import numpy as np
import pandas as pd
import cooler
from tqdm import tqdm, trange

def parse_args():
	parser = argparse.ArgumentParser(description="Higashi visualization tool")
	parser.add_argument('-c', '--config', type=str, default="")
	parser.add_argument('-n', '--neighbor', default=False, action='store_true')
	
	return parser.parse_args()


def skip_start_end(config, chrom="chr1"):
	res = config['resolution']
	gap_tab = pd.read_table(config["cytoband_path"], sep="\t", header=None)
	gap_tab.columns = ['chrom', 'start', 'end', 'sth', 'type']
	gap_list = gap_tab[(gap_tab["chrom"] == chrom) & (gap_tab["type"] == "acen")]
	start = np.floor((np.array(gap_list['start']) - 100000) / res).astype('int')
	end = np.ceil((np.array(gap_list['end']) + 100000) / res).astype('int')
	
	
	return start, end



class HigashiDict(dict):
	def __init__(self, chrom2info, cell_list, chrom_list, **args):
		super().__init__(**args)
		
		self.chrom2info = chrom2info
		self.cell_list = cell_list
		self.chrom_list = chrom_list
		for cell in cell_list:
			self.__setitem__(cell, 1)
		
		
	def keys(self):
		return self.cell_list
	
	def __getitem__(self, key):
		x_all, y_all, count_all = [], [],  []
		
		for chrom in self.chrom_list:
			size, mask_start, mask_end,  impute_f, xs, ys, m1, off_set = self.chrom2info[chrom]
			v = np.array(impute_f[key])
			x_all.append(xs + off_set)
			y_all.append(ys + off_set)
			count_all.append(v.reshape((-1)))
		x_all = np.concatenate(x_all, axis=0).reshape((-1))
		y_all = np.concatenate(y_all, axis=0).reshape((-1))
		count_all = np.concatenate(count_all, axis=0).reshape((-1))

		tab = pd.DataFrame({'bin1_id':x_all,
		                     'bin2_id': y_all,
		                     'count': count_all})

		return tab
	

if __name__ == '__main__':
	args = parse_args()
	config = get_config(args.config)
	res = config['resolution']
	temp_dir = config['temp_dir']
	raw_dir = os.path.join(temp_dir, "raw")
	neighbor_num = config['neighbor_num']
	embedding_name = config['embedding_name']
	
	chrom_list = config['impute_list']
	
	chrom2info = {}
	
	bin_start = 0
	off_set = 0
	
	bins_chrom = []
	bins_start = []
	bins_end = []
	
	cell_list = []
	for chrom_index, chrom in enumerate(chrom_list):
		if args.neighbor:
			impute_f = h5py.File(
				os.path.join(temp_dir, "%s_%s_nbr_%d_impute.hdf5" % (chrom, embedding_name, neighbor_num)),
				"r")
		else:
			impute_f = h5py.File(os.path.join(temp_dir, "%s_%s_nbr_0_impute.hdf5" % (chrom, embedding_name)), "r")
			
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
				
			
	bins = pd.DataFrame({'chrom':bins_chrom, 'start': np.concatenate(bins_start), 'end': np.concatenate(bins_end)})
	cell_name_pixels_dict = HigashiDict(chrom2info, cell_list, chrom_list)
	
	for key in cell_name_pixels_dict:
		print (key)
	print ("Start creating scool")
	
	'''
	cooler.create_scool(cool_uri, bins, cell_name_pixels_dict, columns=None, dtypes=None, metadata=None, assembly=None,
	ordered=False, symmetric_upper=True, mode='w', mergebuf=20000000, delete_temp=True, temp_dir=None, max_merge=200,
	boundscheck=True, dupcheck=True, triucheck=True, ensure_sorted=False, h5opts=None, lock=None, **kwargs)[source]
	Create a single-cell (scool) file.
	
	For each cell store a cooler matrix under /cells, where all matrices have the same dimensions.
	
	Each cell is a regular cooler data collection, so the input must be a bin table and pixel table for each cell.
	The pixel tables are provided as a dictionary where the key is a unique cell name.
	The bin tables can be provided as a dict with the same keys or a single common bin table can be given.
	
	New in version 0.8.9.
	
	Parameters:
	cool_uri (str) – Path to scool file or URI string. If the file does not exist, it will be created.
	bins (pandas.DataFrame or Dict[str, DataFrame]) – A single bin table or dictionary of cell names to bins tables.
	 A bin table is a dataframe with columns chrom, start and end. May contain additional columns.
	 
	cell_name_pixels_dict (Dict[str, DataFrame]) – Cell name as key and pixel table DataFrame as value.
	A table, given as a dataframe or a column-oriented dict, containing columns labeled bin1_id, bin2_id and count, sorted by (bin1_id, bin2_id).
	If additional columns are included in the pixel table, their names and dtypes must be specified using the columns and dtypes arguments.
	For larger input data, an iterable can be provided that yields the pixel data as a sequence of chunks. If the input is a dask DataFrame,
	it will also be processed one chunk at a time.
	
	columns (sequence of str, optional) – Customize which value columns from the input pixels to store in the cooler.
	Non-standard value columns will be given dtype float64 unless overriden using the dtypes argument.
	If None, we only attempt to store a value column named "count".
	
	dtypes (dict, optional) – Dictionary mapping column names to dtypes.
	Can be used to override the default dtypes of bin1_id, bin2_id or count or assign dtypes to custom value columns.
	Non-standard value columns given in dtypes must also be provided in the columns argument or they will be ignored.
	metadata (dict, optional) – Experiment metadata to store in the file. Must be JSON compatible.
	assembly (str, optional) – Name of genome assembly.
	ordered (bool, optional [default: False]) – If the input chunks of pixels are provided with correct triangularity
	and in ascending order of (bin1_id, bin2_id), set this to True to write the cooler in one step. If False (default),
	we create the cooler in two steps using an external sort mechanism. See Notes for more details.
	symmetric_upper (bool, optional [default: True]) – If True, sets the file’s storage-mode property to symmetric-upper:
	use this only if the input data references the upper triangle of a symmetric matrix! For all other cases, set this option to False.
	mode ({'w' , 'a'}, optional [default: 'w']) – Write mode for the output file.
	‘a’: if the output file exists, append the new cooler to it.
	‘w’: if the output file exists, it will be truncated. Default is ‘w’.
	'''
	
	
	cooler.create_scool(os.path.join(temp_dir, "nbr_%d_impute.scool" % (neighbor_num if args.neighbor else 0)), bins, cell_name_pixels_dict, dtypes={'count': 'float32'}, ordered=True)