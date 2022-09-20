import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr, pearsonr, zscore
try:
	from scipy.stats import PearsonRConstantInputWarning, SpearmanRConstantInputWarning
except:
	from scipy.stats import ConstantInputWarning as PearsonRConstantInputWarning

import warnings
from tqdm import trange, tqdm
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from scipy.signal import argrelextrema
cpu_num = multiprocessing.cpu_count()
import scipy.sparse as sps
from sklearn.preprocessing import QuantileTransformer


def smooth(x, window_len=11, window='hanning'):
	"""smooth the data using a window with requested size.

	This method is based on the convolution of a scaled window with the signal.
	The signal is prepared by introducing reflected copies of the signal
	(with the window size) in both ends so that transient parts are minimized
	in the begining and end part of the output signal.

	input:
		x: the input signal
		window_len: the dimension of the smoothing window; should be an odd integer
		window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
			flat window will produce a moving average smoothing.

	output:
		the smoothed signal

	example:

	t=linspace(-2,2,0.1)
	x=sin(t)+randn(len(t))*0.1
	y=smooth(x)

	see also:

	np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
	scipy.signal.lfilter

	TODO: the window parameter could be the window itself if an array instead of a string
	NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
	"""
	x = np.copy(x)
	if x.ndim != 1:
		print("smooth only accepts 1 dimension arrays.")
		raise EOFError
	
	if x.size < window_len:
		print("Input vector needs to be bigger than window size.")
		raise EOFError
	
	if window_len < 3:
		return x
	
	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
		print("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
		raise EOFError
	
	s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
	# print(len(s))
	if window == 'flat':  # moving average
		w = np.ones(window_len, 'd')
	else:
		w = eval('np.' + window + '(window_len)')
	
	y = np.convolve(w / w.sum(), s, mode='valid')
	return y

def pearson_score(m1, m2):
	return pearsonr(m1.reshape((-1)), m2.reshape((-1)))[0]

def spearman_score(m1, m2):
	return spearmanr(m1.reshape((-1)), m2.reshape((-1)))[0]

def vstrans(d1, d2):
	"""
	Variance stabilizing transformation to normalize read counts before computing
	stratum correlation. This normalizes counts so that different strata share similar
	dynamic ranges.
	Parameters
	----------
	d1 : numpy.array of floats
		Diagonal of the first matrix.
	d2 : numpy.array of floats
		Diagonal of the second matrix.
	Returns
	-------
	r2k : numpy.array of floats
		Array of weights to use to normalize counts.
	"""
	# Get ranks of counts in diagonal
	ranks_1 = np.argsort(d1) + 1
	ranks_2 = np.argsort(d2) + 1
	# Scale ranks betweeen 0 and 1
	nranks_1 = ranks_1 / max(ranks_1)
	nranks_2 = ranks_2 / max(ranks_2)
	nk = len(ranks_1)
	r2k = np.sqrt(np.var(nranks_1 / nk) * np.var(nranks_2 / nk))
	return r2k

def global_pearson(mat1, mat2, **kwargs):
	return np.array([pearsonr(mat1.reshape((-1)), mat2.reshape((-1)))[0]])


def global_spearman(mat1, mat2, **kwargs):
	return np.array([spearmanr(mat1.reshape((-1)), mat2.reshape((-1)))[0]])

def pc1_pearson(mat1, mat2):
	pc1 = compartment(mat1).reshape((-1))
	pc2 = compartment(mat2).reshape((-1))
	with warnings.catch_warnings():
		warnings.filterwarnings(
			"ignore"
		)
		s = pearsonr(pc1, pc2)[0]
	return  np.array([s])

def scc_pearson_nonzero(mat1, mat2, max_bins):
	if max_bins < 0:
		max_bins = int(mat1.shape[0]-1)
	if max_bins >= int(mat1.shape[0]-1):
		max_bins = int(mat1.shape[0] - 5)
	
	corr_diag = np.zeros(len(range(max_bins)))
	for d in range(max_bins):
		d1 = mat1.diagonal(d)
		d2 = mat2.diagonal(d)
		# Silence NaN warnings: this happens for empty diagonals and will
		# not be used in the end.
		d1 = d1[d2 > 0]
		d2 = d2[d2 > 0]
		if (d1 == d1[0]).all() or (d2 == d2[0]).all():
			corr_diag[d] = np.nan
		elif len(d2) < 5:
			corr_diag[d] = np.nan
		else:
			with warnings.catch_warnings():
				warnings.filterwarnings(
					"ignore"
				)
				# Compute raw pearson coeff for this diag
				# corr_diag[d] = ss.pearsonr(d1, d2)[0]
				# d1 = zscore(d1)
				# d2 = zscore(d2)
				corr_diag[d] = pearsonr(d1, d2)[0]
	return corr_diag[1:]


def scc_spearman_nonzero(mat1, mat2, max_bins):
	if max_bins < 0:
		max_bins = int(mat1.shape[0] - 5)
	
	corr_diag = np.zeros(len(range(max_bins)))
	for d in range(max_bins):
		d1 = mat1.diagonal(d)
		d2 = mat2.diagonal(d)
		# Silence NaN warnings: this happens for empty diagonals and will
		# not be used in the end.
		d1 = d1[d2 > 0]
		d2 = d2[d2 > 0]
		if (d1 == d1[0]).all() or (d2 == d2[0]).all():
			corr_diag[d] = np.nan
		elif len(d2) < 5:
			corr_diag[d] = np.nan
		else:
			with warnings.catch_warnings():
				warnings.filterwarnings(
					"ignore", category=SpearmanRConstantInputWarning)
				# Compute raw pearson coeff for this diag
				# corr_diag[d] = ss.pearsonr(d1, d2)[0]
				corr_diag[d] = spearmanr(d1, d2)[0]
	return corr_diag[1:]

def scc_pearson(mat1, mat2, max_bins):
	if max_bins < 0:
		max_bins = int(mat1.shape[0] - 5)
	
	corr_diag = np.zeros(len(range(max_bins)))
	for d in range(max_bins):
		d1 = mat1.diagonal(d)
		d2 = mat2.diagonal(d)
		# Silence NaN warnings: this happens for empty diagonals and will
		# not be used in the end.
		
		if (d1 == d1[0]).all() or (d2 == d2[0]).all():
			corr_diag[d] = np.nan
		elif len(d2) < 5:
			corr_diag[d] = np.nan
		else:
			with warnings.catch_warnings():
				warnings.filterwarnings(
					"ignore", category=PearsonRConstantInputWarning
				)
				# Compute raw pearson coeff for this diag
				# corr_diag[d] = ss.pearsonr(d1, d2)[0]
				# d1 = zscore(d1)
				# d2 = zscore(d2)
				corr_diag[d] = pearsonr(d1, d2)[0]
	return corr_diag[1:]


def scc_spearman(mat1, mat2, max_bins):
	if max_bins < 0:
		max_bins = int(mat1.shape[0] - 5)
	
	
	corr_diag = np.zeros(len(range(max_bins)))
	for d in range(max_bins):
		d1 = mat1.diagonal(d)
		d2 = mat2.diagonal(d)
		# Silence NaN warnings: this happens for empty diagonals and will
		# not be used in the end.
		
		if (d1 == d1[0]).all() or (d2 == d2[0]).all():
			corr_diag[d] = np.nan
		elif len(d2) < 5:
			corr_diag[d] = np.nan
		else:
			with warnings.catch_warnings():
				warnings.filterwarnings(
					"ignore", category=SpearmanRConstantInputWarning
				)
				# Compute raw pearson coeff for this diag
				# corr_diag[d] = ss.pearsonr(d1, d2)[0]
				corr_diag[d] = spearmanr(d1, d2)[0]
	return corr_diag[1:]


def get_scc(mat1, mat2, max_bins):
	corrs, weights = [], []
	if max_bins < 0:
		max_bins = int(mat1.shape[0] - 5)
	mat1 = csr_matrix(mat1)
	mat2 = csr_matrix(mat2)
	for stratum in range(max_bins):
		s1 = mat1.diagonal(stratum)
		s2 = mat2.diagonal(stratum)
		mask = (~np.isnan(s1)) & (~np.isnan(s2))
		s1 = s1[mask]
		s2 = s2[mask]
		if (s1 == s1[0]).all() or (s2 == s2[0]).all():
			weights.append(0)
			corrs.append(0)
		elif np.var(s1) == 0 or np.var(s2) == 0:
			weights.append(0)
			corrs.append(0)
		else:
			# zero_pos = [k for k in range(len(s1)) if s1[k] == 0 and s2[k] == 0]
			# s1, s2 = np.delete(s1, zero_pos), np.delete(s2, zero_pos)
			weights.append(len(s1) * np.std(s1) * np.std(s2))
			corrs.append(np.corrcoef(s1, s2)[0, 1])
	corrs = np.nan_to_num(corrs)
	s = np.inner(corrs, weights) / (np.sum(weights))
	return s

def get_scc2(mat1, mat2, max_bins):
	"""
	Compute the stratum-adjusted correlation coefficient (SCC) between two
	Hi-C matrices up to max_dist. A Pearson correlation coefficient is computed
	for each diagonal in the range of 0 to max_dist and a weighted sum of those
	coefficients is returned.
	Parameters
	----------
	mat1 : scipy.sparse.csr_matrix
		First matrix to compare.
	mat2 : scipy.sparse.csr_matrix
		Second matrix to compare.
	max_bins : int
		Maximum distance at which to consider, in bins.
	Returns
	-------
	scc : float
		Stratum adjusted correlation coefficient.
	"""
	
	
	if max_bins < 0 or max_bins > int(mat1.shape[0] - 5):
		max_bins = int(mat1.shape[0] - 5)
	
	
	mat1 = csr_matrix(mat1)
	mat2 = csr_matrix(mat2)
	corr_diag = np.zeros(len(range(max_bins)))
	weight_diag = corr_diag.copy()
	for d in range(max_bins):
		d1 = mat1.diagonal(d)
		d2 = mat2.diagonal(d)
		mask = (~np.isnan(d1)) & (~np.isnan(d2))
		d1 = d1[mask]
		d2 = d2[mask]
		# Silence NaN warnings: this happens for empty diagonals and will
		# not be used in the end.
		with warnings.catch_warnings():
			warnings.filterwarnings(
				"ignore", category=PearsonRConstantInputWarning
			)
			# Compute raw pearson coeff for this diag
			# corr_diag[d] = ss.pearsonr(d1, d2)[0]
			cor = pearsonr(d1, d2)[0]
			corr_diag[d] = cor
			# corr_diag[d] = spearmanr(d1, d2)[0]
		# Compute weight for this diag
		r2k = vstrans(d1, d2)
		weight_diag[d] = len(d1) * r2k
	
	corr_diag, weight_diag = corr_diag[1:], weight_diag[1:]
	mask = ~np.isnan(corr_diag)
	corr_diag, weight_diag = corr_diag[mask], weight_diag[mask]
	# Normalize weights
	
	weight_diag /= sum(weight_diag)
	# Weighted sum of coefficients to get SCCs
	scc = np.nansum(corr_diag * weight_diag)
	return scc, max_bins - np.sum(mask)


def dropcols_coo(M, idx_to_drop):
	idx_to_drop = np.unique(idx_to_drop)
	C = M.tocoo()
	keep = ~np.in1d(C.col, idx_to_drop)
	C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
	C.col -= idx_to_drop.searchsorted(C.col)  # decrement column indices
	C._shape = (C.shape[0], C.shape[1] - len(idx_to_drop))
	return C.tocsr()


def removeRowCSR(mat, i):
	if not isinstance(mat, sps.csr_matrix):
		raise ValueError("works only for CSR format -- use .tocsr() first")
	n = mat.indptr[i + 1] - mat.indptr[i]
	if n > 0:
		mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i + 1]:]
		mat.data = mat.data[:-n]
		mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i + 1]:]
		mat.indices = mat.indices[:-n]
	mat.indptr[i:-1] = mat.indptr[i + 1:]
	mat.indptr[i:] -= n
	mat.indptr = mat.indptr[:-1]
	mat._shape = (mat._shape[0] - 1, mat._shape[1])


def removeZeroDiagonalCSR(mtx, i=0, toRemovePre=None):
	iteration = 0
	toRemove = []
	ctr = 0

	if toRemovePre is not None:
		for items in toRemovePre:
			toRemove.append(items)

	if i == 0:
		diagonal = mtx.diagonal()
		#        print diagonal
		for values in diagonal:
			if values == 0:
				toRemove.append(ctr)
			ctr += 1

	else:
		rowSums = mtx.sum(axis=0)
		rowSums = list(np.array(rowSums).reshape(-1, ))
		rowSums = list(enumerate(rowSums))
		for value in rowSums:
			if int(value[1]) == 0:
				toRemove.append(value[0])
				rowSums.remove(value)
		rowSums.sort(key=lambda tup: tup[1])
		size = len(rowSums)
		perc = i / 100.0
		rem = int(perc * size)
		while ctr < rem:
			toRemove.append(rowSums[ctr][0])
			ctr += 1
	list(set(toRemove))
	toRemove.sort()
	# print toRemove
	mtx = dropcols_coo(mtx, toRemove)
	for num in toRemove:
		if iteration != 0:
			num -= iteration
		removeRowCSR(mtx, num)
		iteration += 1
	return [mtx, toRemove]

def knightRuizAlg(A, tol=1e-8, f1=False):
	n = A.shape[0]
	e = np.ones((n, 1), dtype=np.float64)
	res = []

	Delta = 3
	delta = 0.1
	x0 = np.copy(e)
	g = 0.9

	etamax = eta = 0.1
	stop_tol = tol * 0.5
	x = np.copy(x0)

	rt = tol ** 2.0
	v = x * (A.dot(x))
	rk = 1.0 - v
	#    rho_km1 = np.dot(rk.T, rk)[0, 0]
	rho_km1 = ((rk.transpose()).dot(rk))[0, 0]
	rho_km2 = rho_km1
	rout = rold = rho_km1

	MVP = 0  # we'll count matrix vector products
	i = 0  # outer iteration count

	if f1:
		print("it        in. it      res\n"),
	k = 0
	while rout > rt:  # outer iteration
		i += 1

		if i > 30:
			break

		k = 0
		y = np.copy(e)
		innertol = max(eta ** 2.0 * rout, rt)

		while rho_km1 > innertol:  # inner iteration by CG
			k += 1
			if k == 1:
				Z = rk / v
				p = np.copy(Z)
				# rho_km1 = np.dot(rk.T, Z)
				rho_km1 = (rk.transpose()).dot(Z)
			else:
				beta = rho_km1 / rho_km2
				p = Z + beta * p

			if k > 10:
				break

			# update search direction efficiently
			w = x * A.dot(x * p) + v * p
			# alpha = rho_km1 / np.dot(p.T, w)[0,0]
			alpha = rho_km1 / (((p.transpose()).dot(w))[0, 0])
			ap = alpha * p
			# test distance to boundary of cone
			ynew = y + ap

			if np.amin(ynew) <= delta:

				if delta == 0:
					break

				ind = np.where(ap < 0.0)[0]
				gamma = np.amin((delta - y[ind]) / ap[ind])
				y += gamma * ap
				break

			if np.amax(ynew) >= Delta:
				ind = np.where(ynew > Delta)[0]
				gamma = np.amin((Delta - y[ind]) / ap[ind])
				y += gamma * ap
				break

			y = np.copy(ynew)
			rk -= alpha * w
			rho_km2 = rho_km1
			Z = rk / v
			# rho_km1 = np.dot(rk.T, Z)[0,0]
			rho_km1 = ((rk.transpose()).dot(Z))[0, 0]
		x *= y
		v = x * (A.dot(x))
		rk = 1.0 - v
		# rho_km1 = np.dot(rk.T, rk)[0,0]
		rho_km1 = ((rk.transpose()).dot(rk))[0, 0]
		rout = rho_km1
		MVP += k + 1

		# update inner iteration stopping criterion
		rat = rout / rold
		rold = rout
		res_norm = rout ** 0.5
		eta_o = eta
		eta = g * rat
		if g * eta_o ** 2.0 > 0.1:
			eta = max(eta, g * eta_o ** 2.0)
		eta = max(min(eta, etamax), stop_tol / res_norm)
		if f1:
			print("%03i %06i %03.3f %e %e \n") % \
			(i, k, res_norm, rt, rout),
			res.append(res_norm)
	if f1:
		print("Matrix - vector products = %06i\n") % \
		(MVP),

	# X = np.diag(x[:,0])
	# x = X.dot(A.dot(X))
	return [x, i, k]

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
			expect = np.sum(diag) / (np.sum(diag != 0.0) + 1e-15)
		if expect == 0:
			new_matrix[rows, cols] = 0.0
		else:
			new_matrix[rows, cols] = diag / (expect)
	new_matrix = new_matrix + new_matrix.T
	return new_matrix

def pearson(matrix):
	return np.corrcoef(matrix)

def compartment(matrix, return_PCA=False, model=None, expected = None):
	contact = matrix
	# np.fill_diagonal(contact, np.max(contact))
	# contact = KRnormalize(matrix)
	# contact[np.isnan(contact)] = 0.0
	contact = sqrt_norm(matrix)
	contact = oe(contact, expected)
	np.fill_diagonal(contact, 1)
	with warnings.catch_warnings():
		warnings.filterwarnings(
			"ignore", category=PearsonRConstantInputWarning
		)
		contact = pearson(contact)
	np.fill_diagonal(contact, 1)
	contact[np.isnan(contact)] = 0.0
	if model is not None:
		y = model.transform(contact)
	else:
		pca = PCA(n_components=1)
		y = pca.fit_transform(contact)
	if return_PCA:
		return y, pca
	else:
		return y

def KRnormalize(contact):
	rawMatrix = sps.csr_matrix(contact)
	mtxAndRemoved = removeZeroDiagonalCSR(rawMatrix, toRemovePre=None)
	rawMatrix = mtxAndRemoved[0]
	# print ("normalize", contact, "zero removed", rawMatrix)
	result = knightRuizAlg(rawMatrix)
	colVec = result[0]
	x = sps.diags(colVec.flatten(), 0, format='csr')
	normalizedMatrix = x.dot(rawMatrix.dot(x))
	contact = np.array(normalizedMatrix.todense())
	return contact


def zscore_norm(matrix):
	v = matrix.reshape((-1))
	if not (v == v[0]).all():
		matrix = zscore(v).reshape((len(matrix), -1))
	return matrix

def sqrt_norm(matrix):
	coverage = (np.sqrt(np.sum(matrix, axis=-1)))
	with np.errstate(divide='ignore', invalid='ignore'):
		matrix = matrix / coverage.reshape((-1, 1))
		matrix = matrix / coverage.reshape((1, -1))
	matrix[np.isnan(matrix)] = 0.0
	matrix[np.isinf(matrix)] = 0.0
	return matrix

def pass_norm(matrix):
	return matrix

def log2_norm(matrix):
	return np.log2(1+np.abs(matrix)) * np.sign(matrix)

def log10_norm(matrix):
	return np.log10(1+np.abs(matrix)) * np.sign(matrix)

def quantile_norm(matrix,n_q=250, dist='uniform', clipping=None):
	if len(matrix.shape) == 2:
		matrix[~np.isnan(matrix)] = QuantileTransformer(n_quantiles=n_q, output_distribution=dist).fit_transform(matrix[~np.isnan(matrix)].reshape((-1, 1))).reshape((-1))#.reshape((len(matrix), -1))
	else:
		matrix = QuantileTransformer(n_quantiles=n_q, output_distribution=dist).fit_transform(matrix)
		
	if clipping is not None:
		matrix[matrix > clipping] = clipping
		matrix[matrix < -clipping] = -clipping
	return matrix

