import os
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore")
from scipy.sparse import csr_matrix
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import h5py
import seaborn as sns
from datetime import datetime
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from functools import partial
from random import shuffle
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from bokeh.layouts import row, column
from bokeh.plotting import curdoc, figure, ColumnDataSource
from bokeh.models.widgets import Slider, Select, Button, Div, PreText, Toggle
from bokeh.models import ColorBar, LinearColorMapper, BasicTicker,  BoxSelectTool, LassoSelectTool, LabelSet, HoverTool, TapTool, WheelZoomTool
from bokeh.models.callbacks import CustomJS
from bokeh.palettes import *
from bokeh.transform import linear_cmap
from bokeh import events

from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.preprocessing import QuantileTransformer

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
try:
	from cachetools import LRUCache
	cache_flag = True
except:
	cache_flag = False
try:
	import cmocean
	cmocean_flag = True
except:
	cmocean_flag = False

def kth_diag_indices(a, k):
	rows, cols = np.diag_indices_from(a)
	if k < 0:
		return rows[-k:], cols[:k]
	elif k > 0:
		return rows[:-k], cols[k:]
	else:
		return rows, cols

def oe(matrix, expected = None):
	new_matrix = np.zeros_like(matrix)
	for k in range(len(matrix)):
		rows, cols = kth_diag_indices(matrix, k)
		diag = np.diag(matrix,k)
		if expected is not None:
			expect = expected[k]
		else:
			expect = np.nanmean(diag)
		if expect == 0:
			new_matrix[rows, cols] = 0.0
		else:
			new_matrix[rows, cols] = diag / (expect+1e-15)
	new_matrix = new_matrix + new_matrix.T
	return new_matrix

def pearson(matrix):
	return np.corrcoef(matrix)

def get_config(config_path = "./config.jSON"):
	c = open(config_path,"r")
	return json.load(c)

def create_mask(k=30):
	global config, mask, origin_sparse
	timestr = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
	msg_list.append("%s - First heatmap on this chromosome, indexing" % timestr)
	format_message()
	try:
		final = np.array(np.sum(origin_sparse, axis=0).todense())
	except:
		final = np.array(np.sum(origin_sparse, axis=0))
	size = origin_sparse[0].shape[-1]
	a = np.zeros((size, size))
	if k > 0:
		for i in range(min(k,len(a))):
			for j in range(len(a) - i):
				a[j, j + i] = 1
				a[j + i, j] = 1
		a = np.ones_like((a)) - a
	
	gap = np.nansum(final, axis=-1, keepdims=False) == 0
	if 'cytoband_path' in config:
		gap_tab = pd.read_table(config['cytoband_path'], sep="\t", header=None)
		gap_tab.columns = ['chrom','start','end','sth', 'type']
		gap_list = gap_tab[(gap_tab["chrom"] == chrom_selector.value) & (gap_tab["type"] == "acen")]
		start = np.floor((np.array(gap_list['start']) - 1000000) / config['resolution']).astype('int')
		end = np.ceil((np.array(gap_list['end']) + 1000000) / config['resolution']).astype('int')
		
		for s,e in zip(start, end):
			a[s:e, :] = 1
			a[:, s:e] = 1
	a[gap, :] = 1
	a[:, gap] = 1
	
	matrix = final
	return a, matrix


def plot_heatmap_RdBu_tad(matrix, normalize=True, cbar=False, cmap=None, force_vmin=None):
	global mask
	fig = plt.figure(figsize=(8, 8))
	if not cbar:
		plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
	
	
	if type(matrix) is not np.ndarray:
		matrix = np.array(matrix.todense())
	
	if np.nansum(matrix > 0) == 0:
		return white_img
	
	if VC_button.active:
		coverage = (np.sqrt(np.nansum(matrix, axis=-1)) + 1e-8)
		matrix = matrix / coverage.reshape((-1, 1))
		matrix = matrix / coverage.reshape((1, -1))
	
	if tad_button.active:
		matrix = oe(matrix)
		matrix = pearson(matrix)
	
	matrix = matrix[matrix_start_slider_y.value:matrix_end_slider_y.value,
			 matrix_start_slider_x.value:matrix_end_slider_x.value]
	
	matrix *= (1 - mask[matrix_start_slider_y.value:matrix_end_slider_y.value,
			 matrix_start_slider_x.value:matrix_end_slider_x.value])
	mask1 = mask[matrix_start_slider_y.value:matrix_end_slider_y.value,
			matrix_start_slider_x.value:matrix_end_slider_x.value].astype('bool')
	
	
	matrix = np.ma.masked_where(mask1, matrix)
	
	
	# use_rows = np.where(np.nansum(mask1 == 1, axis=0) != len(matrix))[0]
	# matrix = matrix[use_rows, :]
	# matrix = matrix[:, use_rows]
	# mask1 = mask1[use_rows, :]
	# mask1 = mask1[:, use_rows]
	
	matrix = matrix[::-1,:]
	mask1 = mask1[::-1, :]
	cmap = heatmap_color_selector.value

	if quantile_button.active:
		# matrix[~mask1] = QuantileTransformer(n_quantiles=1000, output_distribution='normal').fit_transform(
		# 		# 	matrix[~mask1].reshape((-1, 1))).reshape((-1))
		
		matrix = QuantileTransformer(n_quantiles=5000, output_distribution='normal').fit_transform(
					matrix.reshape((-1, 1))).reshape((len(matrix), -1))
		cutoff = (1 - vmin_vmax_slider.value) / 2
		vmin, vmax = np.quantile(matrix[matrix != 0.0], cutoff), np.quantile(matrix[matrix != 0.0], 1-cutoff)+1e-8
		# ax = sns.heatmap(matrix, cmap=cmap, square=True, mask=mask1, cbar=cbar, vmin=vmin, vmax=vmax)
		ax = plt.Axes(fig, [0., 0., 1., 1.])
		ax.set_axis_off()
		fig.add_axes(ax)
		
		# ax.imshow(matrix, cmap=cmap,interpolation='none', vmin=vmin, vmax=vmax)
	else:
		
		cutoff = (1 - vmin_vmax_slider.value) / 2
		v = matrix[(matrix > 1e-8)]
		vmin, vmax = np.quantile(matrix[(matrix > 1e-8)], cutoff), np.quantile(matrix[(matrix > 1e-8)], 1 - cutoff)+1e-8
		if force_vmin is not None:
			vmin = force_vmin
		if vmax - vmin <= 1e-5:
			vmin = 0.0
		
		# ax = sns.heatmap(matrix, cmap=cmap, square=True, mask=mask1, cbar=cbar, vmin=vmin, vmax=vmax)
		ax = plt.Axes(fig, [0., 0., 1., 1.])
		ax.set_axis_off()
		fig.add_axes(ax)
	print (matrix.shape)
	matrix[matrix <= vmin] = vmin
	matrix[matrix >= vmax] = vmax
	matrix = (matrix - vmin)/ (vmax - vmin + 1e-15)
	matrix = np.ma.masked_where(mask1, matrix)
	cmap = plt.get_cmap(cmap)
	# ax.imshow(matrix, cmap=cmap, interpolation='none',vmin=vmin, vmax=vmax)
	img = cmap(matrix, bytes=True)
	if darkmode_button.active:
		ax.set_facecolor('#20262B')
	# ax.get_xaxis().set_visible(False)
	# ax.get_yaxis().set_visible(False)
	
	#
	# canvas = FigureCanvas(fig)
	# canvas.draw()  # draw the canvas, cache the renderer
	# img = np.array(canvas.renderer.buffer_rgba()).astype('int8')
	
	# if rotation_button.active:
	# 	print ("rotation")
	# 	im1 = Image.fromarray(img, mode='RGBA')
	# 	im1 = im1.rotate(-45, expand=True)
	# 	if darkmode_button.active:
	# 		bg_color = (32, 38, 43)
	# 	else:
	# 		bg_color= (255, 255, 255)
	# 	fff = Image.new('RGBA', im1.size, bg_color)
	# 	im1=Image.composite(im1, fff, im1)
	#
	# 	img = np.asarray(im1)
	#
	img = np.asarray(img.view(dtype=np.uint32).reshape((img.shape[0], -1)))
	plt.close(fig)
	
	return img


def get_neighbor(d, neighbor_num):
	neighbor = np.argsort(d)[:neighbor_num]
	# neighbor_new = neighbor
	neighbor_new = neighbor[d[neighbor] < 1.0]
	return neighbor_new


def async_heatmap11(selected, id):
	try:
		global config, origin_sparse
		temp_dir = config['temp_dir']
		if len(selected) == 0:
			return
		# plot raw
		if len(selected) > 1:
			b = np.sum(origin_sparse[selected], axis=0) / len(selected)
		else:
			b = origin_sparse[selected[0]]
		try:
			b = np.array(b.todense())
		except:
			b = np.array(b)
		print (b)
		b = plot_heatmap_RdBu_tad(b, force_vmin=0)
	except Exception as e:
		print ("raw error", e)
		# msg_list.append("original wrong")
		# img = white_img
		b = 0
	return b, id


def async_heatmap12(selected, id):
	try:
		global config, origin_sparse
		temp_dir = config['temp_dir']
		rw_dir = os.path.join(temp_dir, "rw")
		if len(selected) == 0:
			return
		size = origin_sparse[0].shape[0]
		# b = np.zeros((size, size))
		with h5py.File(os.path.join(rw_dir, "rw_%s.hdf5" % chrom_selector.value), "r") as f:
			coordinates = np.array(f['coordinates']).astype('int')
			xs, ys = coordinates[:, 0], coordinates[:, 1]
			p = 0
			for i in selected:
				proba = np.array(f["cell_%d" % i])
				
				p += proba
			# b[xs, ys] += p
			b = csr_matrix((p, (xs, ys)), shape=(size, size))
			b = b + b.T
		b = plot_heatmap_RdBu_tad(b)
	except Exception as e:
		print("error", e)
		msg_list.append("random_walk wrong")
		# img = white_img
		b = 0
	return b, id


def async_heatmap21(selected, id):
	try:
		global config, origin_sparse
		temp_dir = config['temp_dir']
		embedding_name = config['embedding_name']
		if len(selected) == 0:
			return
		size = origin_sparse[0].shape[0]
		
		# b = np.zeros((size, size))
		with h5py.File(os.path.join(temp_dir, chrom_selector.value +"_"+ embedding_name+"_all.hdf5"), "r") as f:
			coordinates = f['coordinates']
			xs, ys = coordinates[:, 0], coordinates[:, 1]
			p = 0
			for i in selected:
				proba = np.array(f["cell_%d" % i])
				proba [proba <= 1e-5] = 0.0
				p += proba
			
			# b[xs.astype('int'), ys.astype('int')] += proba
			# b = b + b.T
			b = csr_matrix((p, (xs, ys)), shape=(size, size))
			b = b + b.T
		b = plot_heatmap_RdBu_tad(b)
	except Exception as e:
		print(e)
		msg_list.append("all wrong")
		img = white_img
	return img, id


def async_heatmap22(selected, id):
	try:
		global config, origin_sparse, bulk, mask
		temp_dir = config['temp_dir']
		embedding_name = config['embedding_name']
		if len(selected) == 0:
			return
		size = origin_sparse[0].shape[0]
		
		# b = np.zeros((size, size))
		
		with h5py.File(os.path.join(temp_dir, chrom_selector.value +"_"+ embedding_name+"_nbr_0_impute.hdf5"), "r") as f:
			coordinates = f['coordinates']
			xs, ys = coordinates[:, 0], coordinates[:, 1]
			p = 0
			for i in selected:
				proba = np.array(f["cell_%d" % i])
				proba -= np.min(proba)
				proba[proba <= 1e-5] = 0.0
				p += proba
			# b[xs.astype('int'), ys.astype('int')] += p
			# b = b + b.T
			b = csr_matrix((p, (xs, ys)), shape=(size, size))
			b = b + b.T
		# b *= (1 - mask)
		b = plot_heatmap_RdBu_tad(b)
	except Exception as e:
		print(e)
		msg_list.append("sc impute wrong")
		# img = white_img
		b = 0
	return b, id


def async_heatmap31(selected, id):
	try:
		global config, origin_sparse, bulk, mask
		temp_dir = config['temp_dir']
		embedding_name = config['embedding_name']
		neighbor_num = config['neighbor_num'] + 1
		if len(selected) == 0:
			return
		size = origin_sparse[0].shape[0]
		
		# b = np.zeros((size, size))
		
		with h5py.File(os.path.join(temp_dir, "%s_%s_nbr_%d_impute.hdf5" % (chrom_selector.value, embedding_name, neighbor_num-1)),
						  "r") as f:
			coordinates = f['coordinates']
			xs, ys = coordinates[:, 0], coordinates[:, 1]
			p = 0.0
			for i in selected:
				proba = np.array(f["cell_%d" % i])
				proba -= np.min(proba)
				proba[proba <= 1e-5] = 0.0
				p += proba
			# b[xs.astype('int'), ys.astype('int')] += p
			# b = b + b.T
			b = csr_matrix((p, (xs, ys)), shape=(size, size))
			b = b + b.T
		# b *= (1 - mask)
		b = plot_heatmap_RdBu_tad(b)
	except Exception as e:
		print(e)
		msg_list.append("neighbor wrong")
		# img = white_img
		b = 0
	return b, id


async def async_heatmap_all(selected):
	global mask, origin_sparse, render_cache, bulk, config
	max_distance = config['maximum_distance']
	res = config['resolution']
	if max_distance < 0:
		max_bin = 1e5
	else:
		max_bin = int(max_distance / res)
		
	source = [heatmap11_source, heatmap12_source, heatmap22_source, heatmap31_source]
	h_list = [heatmap11, heatmap12, heatmap22, heatmap31]
	
	if len(selected) == 1:
		key_name = "%s_%s_%d_%d_%d_%d" % (
		data_selector.value, chrom_selector.value, selected[0], int(rotation_button.active),
		int(darkmode_button.active), int(plot_distance_selector.value))
	else:
		key_name = "nostore"
		
	# if key_name in render_cache:
	# 	img_list = render_cache[key_name]
	# 	for id, img in enumerate(img_list):
	# 		source[id].data['img'] = [np.asarray(img)]
	# 		h_list[id].title.text = h_list[id].title.text.split(":")[0]
	#
	# else:
	if len(mask) != origin_sparse[0].shape[0]:
		mask, bulk = create_mask(k=max_bin)
	pool = ProcessPoolExecutor(max_workers=5)
	p_list = []
	p_list.append(pool.submit(async_heatmap11, selected, 0))
	p_list.append(pool.submit(async_heatmap12, selected, 1))
	# p_list.append(pool.submit(async_heatmap21, selected, 2))
	p_list.append(pool.submit(async_heatmap22, selected, 2))
	p_list.append(pool.submit(async_heatmap31, selected, 3))
	
	
	#
	img_list = [0] * (len(p_list) + 1)
	
	result  = {}
	for p in as_completed(p_list):
		img, id = p.result()
		print ("finish getting image", id)
		# source[id].data['img'] = [np.asarray(img)]
		result[id] = img
	
		
		if type(img) is not int:
			# print (img, id)
			source[id].data['img'] = [np.asarray(img)]
			# source[id].data['img'] = [np.asarray(plot_heatmap_RdBu_tad(img, force_vmin = 0 if id==0 else None))]
		else:
			source[id].data['img'] = white_img
			
		h_list[id].title.text = h_list[id].title.text.split(":")[0]
		img_list[id] = img
	if key_name != "nostore" and cache_flag:
		render_cache[key_name] = img_list
	pool.shutdown(wait=True)
	return


def update_heatmap(selected):
	# print (r.data_source.data)
	print ("update_heatmap", selected)
	if len(selected) > 0:
		for h in [heatmap11, heatmap12, heatmap22, heatmap31]:
			h.title.text += ":(loading)"
		curdoc().add_next_tick_callback(partial(async_heatmap_all, selected))
	
	return
	
	


def update_scatter(selected):
	if len(selected) == 1 or type(selected) == int:
		selected = selected[0]
		nb = neighbor_info[selected]
		s = np.array(['#3c84b1' for i in range(cell_num)])
		l = ['cell' for i in range(cell_num)]
		s[nb] = '#f6a36a'
		for n in nb:
			l[n] = 'cell neighbor'
		s[selected] = '#c94658'
		l[selected] = 'selected'
	elif (type(selected) == list) or (type(selected) == np.ndarray):
		s = np.array(['#3c84b1' for i in range(cell_num)])
		l = ['cell' for i in range(cell_num)]
		s[np.array(selected)] = '#c94658'
		for se in selected:
			l[se] = 'selected'
	else:
		print ("type error", type(selected))
		raise EOFError
	
	r.data_source.data['color'] = s
	r.data_source.data['legend_info'] = l
	
	# source.patch({'color':[(slice(len(s)), s)],
	# 			'legend_info':[(slice(len(l)), l)]})
	try:
		r.selection_glyph.fill_color = 'color'
	except Exception as e:
		print (e)
		pass
	try:
		r.nonselection_glyph.fill_color = 'color'
	except Exception as e:
		print (e)
		pass
	
	r.glyph.fill_color = 'color'
	
	embed_vis.legend.visible = True
	bar.visible = False
	return

def cell_slider_update(attr, old ,new):
	r.data_source.selected.indices = [new]


def update(attr, old, new):
	if type(new) == list:
		if len(new) == 0:
			try:
				color_update([], [], color_selector.value)
				for h in [heatmap11_source, heatmap12_source,  heatmap22_source, heatmap31_source]:
					h.data['img'] = [white_img]
			except:
				pass
			return
		
			
		elif len(new)  == 1:
			# new = int(new)
			if int(local_selection_slider.value) > 1:
				v = np.stack([r.data_source.data["x"], r.data_source.data["y"]], axis=-1)
				distance = np.nansum((v[new,None,:] - v[:, :])** 2, axis=-1)
				new = np.argsort(distance).reshape((-1))[:local_selection_slider.value].astype('int')
				
			update_heatmap(new)
			update_scatter(new)
		else:
			new = np.array(new).astype('int')
			update_scatter(new)
			update_heatmap(new)
			if categorical_info.visible:
				# categorical mode:
				bar_info, count = np.unique(np.array(r.data_source.data['label_info'])[new], return_counts=True)
				categorical_hh1.data_source.data = dict(x=bar_info,
														top=count)
				
				
			elif continuous_info.visible:
				temp = continuous_h_all.data_source.data["x"]
				width = temp[1] - temp[0]
				hedges_miss = temp - width / 2
				hedges = np.array(list(hedges_miss) + [hedges_miss[-1] + width])
				hhist1, _ = np.histogram(r.data_source.data['label_info'][new], bins=hedges)
				continuous_hh1.data_source.data = dict(x=(hedges[:-1] + hedges[1:]) / 2,
													   top=hhist1)

	elif type(new) == int:
		selected = [new]
		if int(local_selection_slider.value) > 1:
			v = np.stack([r.data_source.data["x"], r.data_source.data["y"]], axis=-1)
			distance = np.nansum((v[new,None,:] - v[:, :])** 2, axis=-1)
			new = np.argsort(distance).reshape((-1))[:local_selection_slider.value].astype('int')
			
			selected = list(new)
		update_scatter(selected)
		update_heatmap(selected)
	
	return


def float_color_update(s, vmin=None, vmax=None):
	s[np.isnan(s)] = 0.0
	hhist, hedges = np.histogram(s, bins=20)
	hzeros = np.zeros(len(hedges) - 1)
	hmax = max(hhist) * 1.1
	categorical_info.visible = False
	continuous_info.visible = True
	
	global blackorwhite
	if blackorwhite == "black":
		color1 = "#1A1C1D"
	else:
		color1 = "white"
	

	continuous_h_all.data_source.data = dict(x=(hedges[:-1] + hedges[1:]) / 2, top=hhist, fill_color = [color1] * len(hhist))
	continuous_hh1.data_source.data = dict(x=(hedges[:-1] + hedges[1:]) / 2, top=hzeros)
	
	width = hedges[1] - hedges[0]
	continuous_h_all.glyph.width = width
	continuous_hh1.glyph.width = width
	continuous_info.x_range.start = hedges[0] - width
	continuous_info.x_range.end = hedges[-1] + width
	
	embed_vis.legend.visible = False
	

	
	# source.patch({'color':[(slice(len(s)), s)],
	#               'legend_info': [(slice(len(s)), s)],
	#               'label_info': [(slice(len(s)), s)]})
	color_s = np.clip(s, a_min=np.quantile(s, 0.05) if vmin is None else vmin, a_max=np.quantile(s, 0.95) if vmax is None else vmax)
	# print (color_s)
	color_s = (color_s - color_s.min()) / (color_s.max()- color_s.min())
	import matplotlib as mpl
	pal2 = mpl.cm.get_cmap('RdBu_r')
	color_s = pal2(color_s)
	print(color_s)
	color_s = np.array([mpl.colors.to_hex(css) for css in color_s])
	print(color_s)
	r.data_source.data['color'] = color_s
	r.data_source.data['legend_info'] = s
	r.data_source.data['label_info'] = s
	
	# mapper = linear_cmap('color', palette=pal, low=np.quantile(s, 0.05) if vmin is None else vmin, high=np.quantile(s, 0.95) if vmax is None else vmax)
	
	
	
	# try:
	# 	r.selection_glyph.fill_color=mapper
	# except:
	# 	pass
	# try:
	# 	r.nonselection_glyph.fill_color=mapper
	# except:
	# 	pass
	r.glyph.fill_color = 'color'
	
	# bar.color_mapper.low = np.quantile(s, 0.05) if vmin is None else vmin
	# bar.color_mapper.high =np.quantile(s, 0.95) if vmax is None else vmax
	bar.update(color_mapper=LinearColorMapper( palette=pal, low=np.quantile(s, 0.05) if vmin is None else vmin, high=np.quantile(s, 0.95) if vmax is None else vmax))
	bar.visible = True
	return
	
def str_color_update(s, new):
	categorical_info.visible=True
	continuous_info.visible=False
	bar_info, count = np.unique(s, return_counts=True)
	global blackorwhite, config
	if blackorwhite == "black":
		color1 = "#1A1C1D"
	else:
		color1="white"
	
	categorical_h_all.data_source.data = dict(x=bar_info, top=count, fill_color=[color1] * len(bar_info))
	categorical_hh1.data_source.data = dict(x=bar_info, top=[0.0] * len(bar_info))
	
	categorical_info.x_range.factors = list(bar_info)
	
	
	bar.visible = False
	
	l, inv = np.unique(s, return_inverse=True)
	
	if 'vis_palette' not in config:
		if len(l) <= 10:
			encoded_color = [Category10_10[xx] for xx in inv]
		elif len(l) <= 50:
			Category20_20_temp = np.array(Category20_20)
			Category20_20_temp = list(Category20_20_temp) + list(np.array(Spectral11)) + list(Category20_20_temp) + list(np.array(Spectral11))
			# Category20_20_temp = list(Category20_20_temp[np.array([0,2,4,6,8,10,12,14,16,18])]) + list(Category20_20_temp[np.array([1,3,5,7,9,11,13,15,17,19])]) +   list(Category20b_20)
			# shuffle(Category20_20_temp)
			encoded_color = [Category20_20_temp[xx] for xx in inv]
		else:
			encoded_color = [Category10_10[0]] * len(s)
	else:
		if new not in config['vis_palette']:
			if len(l) <= 10:
				encoded_color = [Category10_10[xx] for xx in inv]
			elif len(l) <= 50:
				Category20_20_temp = np.array(Category20_20)
				Category20_20_temp = list(Category20_20_temp) + list(Category20_20_temp) + list(Category20_20_temp)
				# Category20_20_temp = list(Category20_20_temp[np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])]) + list(
				# 	Category20_20_temp[np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])])  +   list(Category20b_20)
				# shuffle(Category20_20_temp)
				encoded_color = [Category20_20_temp[xx] for xx in inv]
			else:
				encoded_color = [Category10_10[0]] * len(s)
		else:
			pal1 = config['vis_palette'][new]
			encoded_color = [pal1[xx] for xx in s]
		# s = list(s)
	# source.patch({
	# 	'legend_info': [(slice(len(s)), s)],
	# 	'label_info': [(slice(len(s)), s)],
	# 	'color': [(slice(len(s)), encoded_color)]
	# })
	
	r.data_source.data['label_info'] = s
	r.data_source.data['legend_info'] = s
	if encoded_color is not None:
		r.data_source.data['color'] = encoded_color
		
		embed_vis.legend.visible = True
	else:
		embed_vis.legend.visible = False
		
	try:
		r.selection_glyph.fill_color='color'
	except:
		pass
	try:
		r.nonselection_glyph.fill_color='color'
	except:
		pass
	r.glyph.fill_color = 'color'

	
def color_update(attr, old, new):
	categorical_info.title.text = "%s bar plot" % new
	continuous_info.title.text = "%s histogram" % new
	global origin_sparse
	if new == "None":
		print ("None color update")
		s = ['cell'] * cell_num
		str_color_update(s, "None")
	elif new == "kde":
		v = np.stack([r.data_source.data["x"], r.data_source.data["y"]], axis=-1)
		model1 = gaussian_kde(v.T)
		model1 = gaussian_kde(v.T, bw_method=model1.factor/2)
		s = model1(v.T).reshape((-1))
		
		float_color_update(np.log(s))
	elif new == "kde_ratio":
		v = np.stack([r.data_source.data["x"], r.data_source.data["y"]], axis=-1)
		model1 = gaussian_kde(v.T)
		model2 = gaussian_kde(v.T, bw_method=model1.factor)
		model1 = gaussian_kde(v.T, bw_method=model1.factor/2)
		s = model1(v.T).reshape((-1))
		s2 = model2(v.T).reshape((-1))
		# kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(v)
		# s = kde.score_samples(v)
		# kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(v)
		# s2 = kde.score_samples(v)
		float_color_update(np.log(s)-np.log(s2))
	elif new == 'read_count':
		
		s = np.array([a.sum() for a in origin_sparse])
		print (s)
		if np.max(s) > origin_sparse[0].shape[0]:
			s = np.log10(s+1)
		float_color_update(s)
	elif new == 'cis_trans_ratio':
		s = []
		for a in origin_sparse:
			c = 0
			for i in range(20):
				c += (a.diagonal(i)).sum()
			s.append((c * 2 - (a.diagonal(0)).sum()) / (a).sum())
		
		s = np.array(s)
		float_color_update(s)
	elif new == 'kmeans':
		global config
		temp_dir = config['temp_dir']
		embedding_name = config['embedding_name']
		# generate embedding vectors
		temp_str = "_origin"
		embed_dir = os.path.join(temp_dir, "embed")
		embed = np.load(os.path.join(embed_dir, "%s_0%s.npy" % (embedding_name, temp_str)))
		s = KMeans(n_clusters=4, n_init=200).fit_predict(embed)
		s = ['cl_'+str(_) for _ in s]
		str_color_update(s, new)
	else:
		s = np.array(color_scheme[new])
		if s.dtype == 'int':
			# categorical
			print ("int")
			s = s.astype("str")
			str_color_update(s, new)
		elif s.dtype == '|S3':
			print("string")
			s = np.asarray([sth.decode('utf8') for sth in s]).astype('str')
			str_color_update(s, new)
			
		elif s.dtype == 'float':
			# continuous
			print ("float")
			float_color_update(s)
		else:
			print ("as string")
			s = s.astype('str')
			str_color_update(s, new)


def data_update(attr, old, new):
	print ("data_update")
	global config
	embed_vis.title.text = "Loading... Please wait"
	
	r.data_source.selected.indices = []
	# color_selector.value = "None"
	initialize(name2config[new], correct_color=True, to_none_color=True)
	# print(config['impute_list'])
	chrom_selector.options = config['impute_list']
	chrom_selector.value = config['impute_list'][0]

	
	global origin_sparse
	temp_dir = config['temp_dir']
	raw_dir = os.path.join(temp_dir, "raw")
	origin_sparse = np.load(os.path.join(raw_dir, "%s_sparse_adj.npy" % config['impute_list'][0]), allow_pickle=True)
	categorical_h_all.data_source.selected.indices = []
	matrix_start_slider_x.value=0
	matrix_start_slider_x.end = origin_sparse[0].shape[-1]
	matrix_end_slider_x.end = origin_sparse[0].shape[-1]
	matrix_end_slider_x.value = origin_sparse[0].shape[-1]
	matrix_start_slider_y.value = 0
	matrix_start_slider_y.end = origin_sparse[0].shape[-1]
	matrix_end_slider_y.end = origin_sparse[0].shape[-1]
	matrix_end_slider_y.value = origin_sparse[0].shape[-1]
	plot_distance_selector.value = origin_sparse[0].shape[-1]
	plot_distance_selector.end = origin_sparse[0].shape[-1]
	local_selection_slider.end = origin_sparse[0].shape[-1]
	
def reload_update(button):
	embed_vis.title.text = "Loading... Please wait"
	initialize(name2config[data_selector.value], correct_color=True)
	
	embed_vis.title.text = "%s projection of embeddings" %(dim_reduction_selector.value)


def reduction_update(attr, old, new):
	initialize(name2config[data_selector.value], correct_color=True)
	
	
def widget_update():
	cell_slider.end=cell_num
	keys = list(color_scheme.keys())
	if "cell_name_higashi" in keys:
		keys.remove("cell_name_higashi")
	color_selector.options = ["None"] + keys+ ["kde", "kde_ratio", "read_count", "cis_trans_ratio", 'kmeans']


def mds(mat, n=2):
	"""
	Multidimensional scaling, MDS.

	Parameters
	----------
	mat : numpy.ndarray
		Distance matrix of the data points.

	n : int, optional
		The dimension of the projected points.
		The default is 2.
	Returns
	-------
	co : numpy.ndarray
		Coordinates of the projected points.
	"""
	
	# mat = np.sqrt(2 - 2 * mat)
	h = np.eye(len(mat)) - np.ones(mat.shape) / len(mat)
	k = -0.5 * h.dot(mat * mat).dot(h)
	if np.any(np.isnan(k)):
		k[np.isnan(k)] = 0
	w, v = np.linalg.eig(k)
	max_ = np.argsort(w)[:-n - 1:-1]
	co = np.real(v[:, max_].dot(np.sqrt(np.diag(w[max_]))))
	# co = np.real(v[:, :2].dot(np.sqrt(np.diag(w[:2]))))
	return co

async def calculate_and_update(v, neighbor_num, correct_color, to_none_color):
	global neighbor_info, source, config, color_scheme
	
	
	distance = pairwise_distances(v, metric='euclidean')
	distance_sorted = np.sort(distance, axis=-1)
	distance /= np.quantile(distance_sorted[:, 1:neighbor_num].reshape((-1)), q=0.5)
	info = xy_selector.value.split("-")
	x_selector_value, y_selector_value = int(info[0]), int(info[1])
	
	if dim_reduction_selector.value == "PCA":
		
		v = PCA(n_components=3).fit_transform(v)
		x, y = v[:, int(x_selector_value) - 1], v[:, int(y_selector_value) - 1]
		# print (x, y)
	elif dim_reduction_selector.value == "UMAP":
		if max(int(x_selector_value), int(y_selector_value)) < 3:
			
			model = UMAP(n_components=2)
		else:
			model = UMAP(n_components=3)
		if "UMAP_params" in config:
			params = dict(config['UMAP_params'])
			for key in params:
				setattr(model, key, params[key])
		v = model.fit_transform(v)
		x, y = v[:, int(x_selector_value) - 1], v[:, int(y_selector_value) - 1]
		timestr = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
		msg_list.append("%s - UMAP finished" % timestr)
		format_message()
	elif dim_reduction_selector.value == "TSNE":
		if max(int(x_selector_value), int(y_selector_value)) < 3:
			model = TSNE(n_components=2, n_jobs=-1, init='pca')
		else:
			model = TSNE(n_components=3, n_jobs=-1, init='pca')
		if "TSNE_params" in config:
			params = config['TSNE_params']
			for key in params:
				setattr(model, key, params[key])
		v = model.fit_transform(v)
		x, y = v[:, 0], v[:, 1]
		timestr = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
		msg_list.append("%s - TSNE finished" % timestr)
		format_message()
	elif dim_reduction_selector.value == 'MDS-euclidean':
		v = pairwise_distances(v, metric='euclidean')
		v =  mds(v, 2)
		x, y = v[:, 0], v[:, 1]
		timestr = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
		msg_list.append("%s - MDS finished" % timestr)
		format_message()
	elif dim_reduction_selector.value == 'MDS-cosine':
		v = pairwise_distances(v, metric='cosine')
		
		v = mds(v, 2)
		x, y = v[:, 0], v[:, 1]
		timestr = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
		msg_list.append("%s - MDS finished" % timestr)
		format_message()
		
	# generate neighbor info
	neighbor_info = np.argsort(distance, axis=-1)[:, :neighbor_num]
	# print (x, y)
	data = dict(x=x, y=y, color=["#3c84b1"] * len(x), legend_info=['cell'] * len(x),
				label_info=np.array(['cell'] * len(x)))
	
	if 'cell_name_higashi' in color_scheme:
		data['cell_name'] = color_scheme['cell_name_higashi']
	source.data = data
	# r.data_source
	if correct_color:
		print ("correct color")
		if to_none_color:
			try:
				color_selector.remove_on_change('value', color_update)
			except:
				pass
			color_selector.value = "None"
			color_selector.on_change('value', color_update)
		else:
			color_update([], [], color_selector.value)

		widget_update()
		update([], [], [])
		r.data_source.selected.indices = []
		embed_vis.title.text = "%s projection of embeddings" % (dim_reduction_selector.value)
	print ("finish init")
	return

def initialize(config_name, correct_color=False, to_none_color=False):
	global config, color_scheme, v, cell_num, source, neighbor_info
	config = get_config(config_name)
	temp_dir = config['temp_dir']
	data_dir = config['data_dir']
	embedding_name = config['embedding_name']
	neighbor_num = int(config['neighbor_num'])
	heatmap31.title.text = "Higashi(%d)"  % (neighbor_num)
	color_scheme = {}
	try:
		with open(os.path.join(data_dir, "label_info.pickle"), "rb") as f:
			color_scheme = pickle.load(f)
	except:
		color_scheme = {}
	
	# generate embedding vectors
	temp_str = "_origin"
	embed_dir = os.path.join(temp_dir, "embed")
	v = np.load(os.path.join(embed_dir, "%s_0%s.npy" % (embedding_name, temp_str)))

	cell_num = len(v)
	if dim_reduction_selector.value == "UMAP":
		timestr = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
		msg_list.append("%s - UMAP computing, it takes time" % timestr)
		format_message()
	elif dim_reduction_selector.value == "TSNE":
		timestr = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
		msg_list.append("%s - TSNE computing, it takes time" % timestr)
		format_message()
	elif dim_reduction_selector.value == "MDS-euclidean":
		timestr = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
		msg_list.append("%s - MDS computing, it takes time" % timestr)
		format_message()
	elif dim_reduction_selector.value == "MDS-cosine":
		timestr = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
		msg_list.append("%s - MDS computing, it takes time" % timestr)
		format_message()
	curdoc().add_next_tick_callback(partial(calculate_and_update, v, neighbor_num, correct_color, to_none_color))

	
	
def Kmean_ARI(button):
	global config, source
	temp_dir = config['temp_dir']
	embed_dir = os.path.join(temp_dir, "embed")
	embedding_name = config['embedding_name']

	# generate embedding vectors
	v = np.load(os.path.join(embed_dir, "%s_0_origin.npy" % embedding_name))
	target = np.array(source.data['label_info'])
	target2int = np.zeros_like(target, dtype='int')
	uniques, counts = np.unique(target, return_counts=True)
	
	for i, t in enumerate(uniques):
		target2int[target == t] = i
	# uniques = uniques[counts >= 40]
	pred = KMeans(n_clusters=len(uniques), n_init = 200).fit(v).labels_
	ari1 = adjusted_rand_score(target2int, pred)
	
	# for i in [len(uniques)-1, len(uniques), len(uniques)+1]:
	# 	pred = KMeans(n_clusters=i, n_init=200).fit(v).labels_
	# 	print (adjusted_rand_score(target2int, pred))
	

	pred = AgglomerativeClustering(n_clusters=len(uniques)).fit(v).labels_
	ari3 = adjusted_rand_score(target2int, pred)
	
	
	v = np.stack([r.data_source.data["x"],r.data_source.data["y"]],axis=-1)
	
	pred = KMeans(n_clusters=len(uniques), n_init=200).fit(v).labels_
	ari2 = adjusted_rand_score(target2int, pred)

	pred = AgglomerativeClustering(n_clusters=len(uniques)).fit(v).labels_
	ari4 = adjusted_rand_score(target2int, pred)
	timestr = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
	msg_list.append("%s - (Kmeans)ARI=%.4f, %.4f, (Hie)ARI=%.4f, %.4f" % (timestr, ari1, ari2, ari3, ari4))
	format_message()
	

def format_message():
	m = ""
	for msg in msg_list[:-1]:
		m += '<div>  {}<br></div>'.format(msg)
	m += '<div><b>  {}</b><br></div> '.format(msg_list[-1])
	info_log.text = m

def clear_log(button):
	global msg_list
	msg_list = []
	info_log.text = '<div> <br></div>'
	
def size_update(attr, old, new):
	r.glyph.size = new

def tapcallback(attr, old, new):
	if len(new) == 1:
		selected = categorical_h_all.data_source.data['x'][new[0]]
		selected = np.where(np.array(source.data['label_info']) == selected)[0]
		r.data_source.selected.indices = list(selected)
	if len(new) == 0:
		r.data_source.selected.indices = []
	
	
def plot_distance_update(attr, old, new):
	global mask
	mask, final = create_mask(int(plot_distance_selector.value))
	update_heatmap(r.data_source.selected.indices)


def anything_that_updates_heatmap(attr, old, new):
	if len(r.data_source.selected.indices) == 1:
		new = r.data_source.selected.indices
		if int(local_selection_slider.value) > 1:
			v = np.stack([r.data_source.data["x"], r.data_source.data["y"]], axis=-1)
			distance = np.nansum((v[new, None, :] - v[:, :]) ** 2, axis=-1)
			new = np.argsort(distance).reshape((-1))[:local_selection_slider.value].astype('int')
			update_heatmap(new)
			return
		
	update_heatmap(r.data_source.selected.indices)

def anything_that_updates_heatmap_button(button):
	if len(r.data_source.selected.indices) == 1:
		new = r.data_source.selected.indices
		if int(local_selection_slider.value) > 1:
			v = np.stack([r.data_source.data["x"], r.data_source.data["y"]], axis=-1)
			distance = np.nansum((v[new, None, :] - v[:, :]) ** 2, axis=-1)
			new = np.argsort(distance).reshape((-1))[:local_selection_slider.value].astype('int')
			update_heatmap(new)
			return
	update_heatmap(r.data_source.selected.indices)


def local_selection_update(attr, old, new):
	anything_that_updates_heatmap(attr, old, new)
	update(attr, old, r.data_source.selected.indices)
def minus_cell(button):
	cell_slider.value -= 1
	cell_slider.value_throttled -= 1
	
def plus_cell(button):
	cell_slider.value += 1
	cell_slider.value_throttled += 1

def axis_update(attr, old, new):
	initialize(name2config[data_selector.value], correct_color=True)

def chrom_update(attr, old, new):
	global origin_sparse
	temp_dir = config['temp_dir']
	raw_dir = os.path.join(temp_dir, "raw")
	origin_sparse = np.load(os.path.join(raw_dir, "%s_sparse_adj.npy" % chrom_selector.value), allow_pickle=True)
	matrix_start_slider_x.end = origin_sparse[0].shape[-1]
	matrix_end_slider_x.end = origin_sparse[0].shape[-1]
	matrix_start_slider_x.value = 0
	matrix_end_slider_x.value = origin_sparse[0].shape[-1]
	matrix_start_slider_y.end = origin_sparse[0].shape[-1]
	matrix_end_slider_y.end = origin_sparse[0].shape[-1]
	matrix_start_slider_y.value = 0
	matrix_end_slider_y.value = origin_sparse[0].shape[-1]
	plot_distance_selector.value = origin_sparse[0].shape[-1]
	plot_distance_selector.end = origin_sparse[0].shape[-1]
	# color_update([], [], color_selector.value)
	update_heatmap(r.data_source.selected.indices)
	

# Initializing some global variables

global config, color_scheme, v, cell_num, source, neighbor_info, mask, origin_sparse, render_cache

import argparse
def parse_args():
	parser = argparse.ArgumentParser(description="Higashi vis program")
	parser.add_argument('-c', '--config', type=str, default="../config_dir/visual_config.JSON")
	
	return parser.parse_args()
args = parse_args()

vis_config = get_config(args.config)
config_dir = vis_config['config_list']
avail_data = []
for c in config_dir:
	avail_data.append(get_config(c)["config_name"])

name2config = {n:c for n,c in zip(avail_data, config_dir)}
if cache_flag:
	render_cache = LRUCache(maxsize=20)
else:
	render_cache = False

mask = np.zeros((1,1))
msg_list = ["-- Higashi-vis Log -- "]
source = ColumnDataSource(data = dict(x=[], y=[], color=[], legend_info=[],
				label_info=np.array([])))

# create all the widgets
dim_reduction_selector = Select(title='Vis method', value="PCA", options = ["PCA", "UMAP", "TSNE", "MDS-euclidean", "MDS-cosine"], width=150)

# x_selector = Select(title="x-axis", value="1", options=["1","2","3"], width=96)
# y_selector = Select(title="y-axis", value="2", options=["1","2","3"], width=96)

xy_selector = Select(title="x-axis / y-axis", value="1-2", options=["1-2", "1-3", "2-3"], width=150)

# create the heatmap visualization
heatmap11 = figure(toolbar_location="above",tools="pan, wheel_zoom, reset, save", plot_width=300, plot_height=300,
				 min_border=5,output_backend="webgl", title='raw', active_scroll = "wheel_zoom")
heatmap12 = figure(toolbar_location="above", tools="pan, wheel_zoom, reset, save",plot_width=300, plot_height=300,x_range=heatmap11.x_range, y_range=heatmap11.y_range,
				 min_border=5,output_backend="webgl", title='conv-rwr', active_scroll = "wheel_zoom")

heatmap22 = figure(toolbar_location="above", tools="pan, wheel_zoom, reset, save",plot_width=300, plot_height=300, x_range=heatmap11.x_range, y_range=heatmap11.y_range,
				 min_border=5,output_backend="webgl", title='Higashi(0)', active_scroll = "wheel_zoom")

heatmap31 = figure(toolbar_location="above", tools="pan, wheel_zoom, reset, save",plot_width=300, plot_height=300, x_range=heatmap11.x_range, y_range=heatmap11.y_range,
				 min_border=5,output_backend="webgl", title='Higashi(4)', active_scroll = "wheel_zoom")

for h in [heatmap11, heatmap12, heatmap22, heatmap31]:
	h.xgrid.visible = False
	h.ygrid.visible = False
	h.xaxis.visible = False
	h.yaxis.visible =False

white = np.ones((20,20,4), dtype='int8') * 255
white_img = white.view(dtype=np.uint32).reshape((white.shape[0], -1))
heatmap11_source = ColumnDataSource(data=dict(img=[white_img],x=[0],y=[0],dw=[10], dh=[10]))
heatmap12_source = ColumnDataSource(data=dict(img=[white_img],x=[0],y=[0],dw=[10], dh=[10]))
heatmap22_source = ColumnDataSource(data=dict(img=[white_img],x=[0],y=[0],dw=[10], dh=[10]))
heatmap31_source = ColumnDataSource(data=dict(img=[white_img],x=[0],y=[0],dw=[10], dh=[10]))
h1 = heatmap11.image_rgba(image='img', x='x', y='y',dw='dw',dh='dh', source=heatmap11_source)
h2 = heatmap12.image_rgba(image='img', x='x', y='y',dw='dw',dh='dh', source=heatmap12_source)
h4 = heatmap22.image_rgba(image='img', x='x', y='y',dw='dw',dh='dh', source=heatmap22_source)
h5 = heatmap31.image_rgba(image='img', x='x', y='y',dw='dw',dh='dh', source=heatmap31_source)


initialize(config_dir[0], True)
keys = list(color_scheme.keys())
if "cell_name_higashi" in keys:
	keys.remove("cell_name_higashi")

color_selector = Select(title='color scheme', value="None", options=["None", "Random"]+keys, width=150)



size_selector = Slider(title='scatter size', value=4, start=1, end=20,step=1, width=150)


data_selector = Select(title='scHi-C dataset', value=avail_data[0], options=avail_data, width=150)


chrom_selector = Select(title='chromosome selector', value=config['chrom_list'][0], options=config['chrom_list'], width=150)
cmap_options = ["Reds", "RdBu_r", "viridis"]
if cmocean_flag:
	cmap_options += ["cmo.curl"]
	
heatmap_color_selector = Select(title='heatmap cmap', value="Reds", options=cmap_options, width=150)

reload_button = Button(label="Reload", button_type="success", width=150)
reload_button.on_click(reload_update)

unsup_button = Button(label="Kmeans-ARI", button_type="primary", width=150)
unsup_button.on_click(Kmean_ARI)



clear_button = Button(label="Clear log", button_type="danger", width=150)
clear_button.on_click(clear_log)

minus_button = Button(label="Previous", button_type='primary', width=150)
plus_button = Button(label="Next", button_type='primary', width=150)
minus_button.on_click(minus_cell)
plus_button.on_click(plus_cell)

temp_dir = config['temp_dir']
impute_list = config['impute_list']
raw_dir = os.path.join(temp_dir, "raw")
origin_sparse = np.load(os.path.join(raw_dir, "%s_sparse_adj.npy" % impute_list[0]), allow_pickle=True)


plot_distance_selector = Slider(title='Heatmap distance', value=origin_sparse[0].shape[-1], start=1, end=origin_sparse[0].shape[-1], step=1, width=150, value_throttled=2000)
local_selection_slider = Slider(title='Local selection number', value=1, start=1, end=origin_sparse[0].shape[-1], step=1, width=150, value_throttled=2000)
rotation_button = Toggle(label="Rotate heatmap", button_type="primary", width=150)
rotation_button.on_click(anything_that_updates_heatmap_button)

tad_button = Toggle(label="Compartment", button_type="primary", width=150)
quantile_button = Toggle(label="Quantile_norm", button_type='primary', width=150, active=False)
VC_button = Toggle(label="VC_SQRT", button_type='primary', width=150, active=False)



pal = sns.color_palette('RdBu_r', 32)
pal = pal.as_hex()
TOOLS="pan,wheel_zoom,tap,box_select,lasso_select,reset, save"

# create the scatter plot
embed_vis = figure(tools=TOOLS, plot_width=600, plot_height=600, min_border=5, min_border_right=20,
		   toolbar_location="above",
		   title="PCA projection of embeddings",output_backend="webgl")

embed_vis.xgrid.visible = False
embed_vis.ygrid.visible = False
embed_vis.select(BoxSelectTool).select_every_mousemove = False
embed_vis.select(LassoSelectTool).select_every_mousemove = False

cell_slider = Slider(title='cell selector', value=0, start=0, end=cell_num,step=1, value_throttled=2000)


r = embed_vis.scatter(x="x", y="y", size=size_selector.value, fill_color="color", line_color=None, legend_field="legend_info",
					  alpha=0.9, source=source, nonselection_fill_alpha = 0.9, selection_fill_color="color", nonselection_fill_color="color")
embed_vis.add_tools(HoverTool(tooltips=[("index", "$index"), ("Label", "@legend_info"), ("Name", "@cell_name")]))
embed_vis.legend.location = "bottom_right"

# create the color bar for continuous label
bar = ColorBar(color_mapper=LinearColorMapper(pal, low=0.0, high=1.0),ticker= BasicTicker(),location=(0,0))
bar.visible=False
bar.background_fill_alpha=0.0
embed_vis.add_layout(bar, 'center')


LINE_ARGS = dict(color="#3A5785", line_color=None)
bar_info, count = np.unique(source.data['label_info'], return_counts=True)
categorical_info = figure(toolbar_location=None,x_range=bar_info ,plot_width=300, plot_height=300, min_border=5, output_backend="webgl", title='None bar plot')
categorical_h_all = categorical_info.vbar(x=bar_info, top=count, width=0.40, color=['white'], line_color="#3A5785")
categorical_hh1 = categorical_info.vbar(x=bar_info, top=[0.0] * len(bar_info), width=0.40, alpha=0.5, **LINE_ARGS)
categorical_info.xaxis.major_label_orientation = math.pi/4
categorical_info.add_tools(HoverTool(tooltips=[("Label", "@x"), ("Count", "@top")]))
categorical_info.add_tools(TapTool())
categorical_h_all.data_source.selected.on_change('indices', tapcallback)

continuous_info = figure(toolbar_location=None,plot_width=300, plot_height=300, min_border=5, output_backend="webgl", title='None histogram')
continuous_h_all = continuous_info.vbar(x=[0.0], top=[0.0], width=0.40, color=['white'], line_color="#3A5785")
continuous_hh1 = continuous_info.vbar(x=[0.0], top=[0.0] * len(bar_info), width=0.40, alpha=0.5, **LINE_ARGS)
continuous_info.add_tools(HoverTool(tooltips=[("Bin", "@x"), ("Count", "@top")]))
continuous_info.visible=False


matrix_start_slider_x = Slider(title="Heatmap start: x", value=0, start=0, end=origin_sparse[0].shape[-1], step=1, value_throttled=2000, width=150)
matrix_end_slider_x = Slider(title="Heatmap end: x", value=origin_sparse[0].shape[-1], start=0, end=origin_sparse[0].shape[-1], step=1, value_throttled=2000, width=150)

matrix_start_slider_y = Slider(title="Heatmap start: y", value=0, start=0, end=origin_sparse[0].shape[-1], step=1, value_throttled=2000, width=150)
matrix_end_slider_y = Slider(title="Heatmap end: y", value=origin_sparse[0].shape[-1], start=0, end=origin_sparse[0].shape[-1], step=1, value_throttled=2000, width=150)
vmin_vmax_slider = Slider(title='Vmin/Vmax(% of range)', value=0.95, start=0.0,end=1.0, step=0.01, value_throttled=2000, width=150)



info_log = Div(text="", width = 1200, height = 300, height_policy="fixed",
				   style={'overflow-y':'scroll',
						  'height':'300px',
						  'width':'1200px',
						  'font-family': 'monospace',
						  'font-size': '16px',
						  'border': '2px solid #198EC7',
						  'border-left': '5px solid #198EC7',
						  'page-break-inside': 'avoid',
						  'padding': '1em 1em',
						  'display': 'block',
						  'overscroll-behavior-y': 'contain',
						  'scroll-snap-type': 'y mandatory'
						  }, css_classes = ['div_container'] )


global blackorwhite
blackorwhite="white"
theme_backup = curdoc().theme
from time import sleep
def change_theme(button):
	global blackorwhite
	if blackorwhite == "white":
		curdoc().theme = "dark_minimal"
		blackorwhite = "black"
		for selector in [data_selector, chrom_selector, dim_reduction_selector, color_selector, xy_selector,  heatmap_color_selector]:
			selector.css_classes = ['custom_select']
		
		for slider in [size_selector, vmin_vmax_slider, cell_slider, plot_distance_selector, local_selection_slider]:
			slider.css_classes = ['custom_slider']
		categorical_h_all.data_source.data['fill_color'] = ["#1A1C1D"] * len(categorical_h_all.data_source.data['x'])
		continuous_h_all.data_source.data['fill_color'] = ["#1A1C1D"] * len(continuous_h_all.data_source.data['x'])
	else:
		curdoc().theme = theme_backup
		blackorwhite = "white"
		for selector in [data_selector, chrom_selector, dim_reduction_selector, color_selector, xy_selector,  heatmap_color_selector]:
			selector.css_classes = []
		for slider in [size_selector, vmin_vmax_slider, cell_slider, plot_distance_selector, local_selection_slider]:
			slider.css_classes = []
		categorical_h_all.data_source.data['fill_color'] = ["white"] * len(categorical_h_all.data_source.data['x'])
		continuous_h_all.data_source.data['fill_color'] = ["white"] * len(continuous_h_all.data_source.data['x'])
	
	update_heatmap(r.data_source.selected.indices)
format_message()

darkmode_button = Toggle(label="Dark mode", button_type="primary", width=150)
darkmode_button.js_on_click(CustomJS(args=dict(button=darkmode_button, div=info_log),
									code='''
									if (button.active) {
									document.body.style.backgroundColor = "#16191C";
									document.body.style.color = "#ffffff";
									}
									else {
									document.body.style.backgroundColor = "white";
									document.body.style.color = "black";
									}
									'''))

darkmode_button.on_click(change_theme)
tad_button.on_click(anything_that_updates_heatmap_button)
VC_button.on_click(anything_that_updates_heatmap_button)
quantile_button.on_click(anything_that_updates_heatmap_button)
layout= column(row(
			row(
				column(heatmap11, heatmap22),
				column(heatmap12, heatmap31),
			),
			embed_vis,
			column(
				row(reload_button, unsup_button),
				row(darkmode_button, rotation_button),
				row(tad_button, clear_button),
				row(data_selector, chrom_selector),
				row(dim_reduction_selector, xy_selector),
				row(color_selector, heatmap_color_selector),
				row(size_selector, local_selection_slider),
				row(matrix_start_slider_x, matrix_end_slider_x),
				row(matrix_start_slider_y, matrix_end_slider_y),
				row(plot_distance_selector, vmin_vmax_slider),
				row(quantile_button, VC_button),
				row(minus_button, plus_button),
				row( cell_slider),

			),
		), row(info_log,categorical_info,
				continuous_info))

from bokeh.themes import built_in_themes, Theme

r.data_source.selected.on_change('indices', update)

def release(bar):
	"Build a suitable CustomJS to display the current event in the div model."
	return CustomJS(args=dict(bar=bar),code="""
	bar.indices=[];
	""")



# execute a callback whenever the plot canvas is tapped
embed_vis.js_on_event(events.Tap, release(bar= categorical_h_all.data_source.selected))
color_selector.on_change('value', color_update)
plot_distance_selector.on_change('value_throttled', plot_distance_update)
dim_reduction_selector.on_change('value', reduction_update)
data_selector.on_change('value', data_update)
size_selector.on_change('value', size_update)
chrom_selector.on_change('value', chrom_update)
heatmap_color_selector.on_change('value', anything_that_updates_heatmap)
xy_selector.on_change('value', axis_update)
# y_selector.on_change('value', axis_update)
cell_slider.on_change('value_throttled', cell_slider_update)
vmin_vmax_slider.on_change('value_throttled', anything_that_updates_heatmap)
local_selection_slider.on_change('value_throttled', local_selection_update)
matrix_start_slider_x.on_change('value_throttled', anything_that_updates_heatmap)
matrix_end_slider_x.on_change('value_throttled', anything_that_updates_heatmap)
matrix_start_slider_y.on_change('value_throttled', anything_that_updates_heatmap)
matrix_end_slider_y.on_change('value_throttled', anything_that_updates_heatmap)
curdoc().title = "Higashi-vis"
curdoc().add_root(layout)



