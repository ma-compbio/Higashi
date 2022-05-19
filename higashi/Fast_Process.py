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
	
	
def data2mtx(config, file, chrom_start_end, verbose, cell_id):
	if "header_included" in config:
		if config['header_included']:
			tab = pd.read_table(file, sep="\t")
		else:
			tab = pd.read_table(file, sep="\t", header=None)
			tab.columns = config['contact_header']
	else:
		tab = pd.read_table(file, sep="\t", header=None)
		tab.columns = config['contact_header']
	if 'count' not in tab.columns:
		tab['count'] = 1
	
	if 'downsample' in config:
		downsample = config['downsample']
	else:
		downsample = 1.0
		
	data = tab
	# fetch info from config
	res = config['resolution']
	chrom_list = config['chrom_list']
	
	data = data[(data['chrom1'] == data['chrom2']) & (np.abs(data['pos2'] - data['pos1']) >= 2500)]
	
	pos1 = np.array(data['pos1'])
	pos2 = np.array(data['pos2'])
	bin1 = np.floor(pos1 / res).astype('int')
	bin2 = np.floor(pos2 / res).astype('int')
	
	chrom1, chrom2 = np.array(data['chrom1'].values), np.array(data['chrom2'].values)
	count = np.array(data['count'].values)
	
	if downsample < 1:
		# print ("downsample at", downsample)
		index = np.random.permutation(len(data))[:int(downsample * len(data))]
		count = count[index]
		chrom1 = chrom1[index]
		bin1 = bin1[index]
		bin2 = bin2[index]
		
	del data
	
	m1_list = []
	for i, chrom in enumerate(chrom_list):
		mask = (chrom1 == chrom)
		size = chrom_start_end[i, 1] - chrom_start_end[i, 0]
		temp_weight2 = count[mask]
		m1 = csr_matrix((temp_weight2, (bin1[mask], bin2[mask])), shape=(size, size), dtype='float32')
		m1 = m1 + m1.T
		m1_list.append(m1)
		count = count[~mask]
		bin1 = bin1[~mask]
		bin2 = bin2[~mask]
		chrom1 = chrom1[~mask]
	
	return m1_list, cell_id





# Extra the data.txt table
# Memory consumption re-optimize
def extract_table(config):
	# fetch info from config
	data_dir = config['data_dir']
	temp_dir = config['temp_dir']
	chrom_list = config['chrom_list']
	if 'input_format' in config:
		input_format = config['input_format']
	else:
		input_format = 'higashi_v2'
	
	chrom_start_end = np.load(os.path.join(temp_dir, "chrom_start_end.npy"))
	if input_format == 'higashi_v1':
		print ("Sorry no higashi_v1")
		raise EOFError
			
	elif input_format == 'higashi_v2':
		print ("extracting from filelist.txt")
		with open(os.path.join(data_dir, "filelist.txt"), "r") as f:
			lines = f.readlines()
			filelist = [line.strip() for line in lines]
		bar = trange(len(filelist))
		mtx_all_list = [[0]*len(filelist) for i in range(len(chrom_list))]
		p_list = []
		pool = ProcessPoolExecutor(max_workers=cpu_num)
		for cell_id, file in enumerate(filelist):
			p_list.append(pool.submit(data2mtx, config, file, chrom_start_end, False, cell_id))
		
		
		for p in as_completed(p_list):
			mtx_list, cell_id = p.result()
			for i in range(len(chrom_list)):
				mtx_all_list[i][cell_id] = mtx_list[i]
			bar.update(1)
		bar.close()
		pool.shutdown(wait=True)
		for i in range(len(chrom_list)):
			np.save(os.path.join(temp_dir, "raw", "%s_sparse_adj.npy" % chrom_list[i]), mtx_all_list[i], allow_pickle=True)
		
	else:
		print ("invalid input format")
		raise EOFError
	
	
	
	

	
		

if __name__ == '__main__':
	args = parse_args()
	config = get_config(args.config)
	
	create_dir(config)
	generate_chrom_start_end(config)
	extract_table(config)