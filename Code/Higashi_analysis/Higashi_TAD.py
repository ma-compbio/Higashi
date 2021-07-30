import numpy as np
from scipy.signal import argrelextrema
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm, trange

def insulation_score(m, windowsize=500000, res=10000):
	windowsize_bin = int(windowsize / res)
	score = np.ones((m.shape[0]))
	for i in range(0, m.shape[0]):
		with np.errstate(divide='ignore', invalid='ignore'):
			v = np.sum(m[max(0, i - windowsize_bin): i, i + 1: min(m.shape[0] - 1, i + windowsize_bin + 1)]) / (np.sum(
				m[max(0, i - windowsize_bin):min(m.shape[0], i + windowsize_bin + 1),
				max(0, i - windowsize_bin):min(m.shape[0], i + windowsize_bin + 1)]))
			if np.isnan(v):
				v = 1.0
		
		score[i] = v
	return score


def call_tads(score, windowsize=500000, res=10000):
	windowsize_bin = int(windowsize / res / 2)
	borders = argrelextrema(np.copy(score), np.less, order=windowsize_bin)
	borders =  borders[0]
	border_score = score[borders]
	return borders[(border_score < 0.999) & (border_score > 0)]

# scTAD_distance from j to i
def scTAD_distance(score, cumsum_score, i, j):
	if type(i) == np.ndarray:
		minimum_score = np.array([np.min(score[min(j, i_d):max(i_d, j) + 1]) for i_d in i])
	elif type(j) == np.ndarray:
		minimum_score = np.array([np.min(score[min(i, j_d):max(j_d, i) + 1]) for j_d in j])
	else:
		minimum_score = np.min(score[min(j, i):max(i, j)])
	
	return np.abs(cumsum_score[i] - cumsum_score[j] - (i - j) * minimum_score)


class scTAD_calibrator():
	
	def __init__(self, K, shape, print_identifier=""):
		self.K = K
		self.print_identifier = print_identifier
		self.shape = shape
	
	
	@staticmethod
	def assign(cell, K, shared_boundaries, sc_score, cum_score, c_bound):
		# For all the sc-boundaries that is left to the first boundaris, assign them to it
		sc_assignment = np.ones((len(c_bound)), dtype='int')
		mask = c_bound < shared_boundaries[0]
		sc_assignment[mask] = 0
		search_range = np.ones((K, 2), dtype='int')
		search_range[:, 0] = 100000
		search_range[:, 1] = 0
		# Because the assignment for each sc-boundaries is either the left closest or the right closes, so...
		for j in range(K - 1):
			mask = (c_bound >= shared_boundaries[j]) & (c_bound < shared_boundaries[j + 1])
			range_boundaris = c_bound[mask]
			left_score = scTAD_distance(sc_score, cum_score, shared_boundaries[j], range_boundaris)
			right_score = scTAD_distance(sc_score, cum_score, shared_boundaries[j + 1],
			                             range_boundaris)
			sc_assignment[mask] = np.where(left_score <= right_score, j, j + 1)
		
		# For all the sc-boundaries that is right to the last boundaries, assign them to it
		mask = c_bound >= shared_boundaries[-1]
		sc_assignment[mask] = K - 1
		
		for j in range(K):
			assigned_boundaries = c_bound[sc_assignment == j]
			if len(assigned_boundaries) == 0:
				continue
			search_range[j][0] = np.min(assigned_boundaries)
			search_range[j][1] = np.max(assigned_boundaries)
		
		return cell, sc_assignment, search_range
	
	
	@staticmethod
	def update(n_cell, sc_boundaries, sc_assignment, sc_score, cum_score, j, start, end):
		value = np.zeros((end - start))
		for cell in range(n_cell):
			c_bound = sc_boundaries[cell]
			assigned_boundaries = c_bound[sc_assignment[cell] == j]
			
			for bin in range(end - start):
				value[bin] += np.sum(
					scTAD_distance(sc_score[cell], cum_score[cell], bin + start, assigned_boundaries))
		
		return np.argmin(value) + start, j
	
	def fit_transform(self, sc_score, sc_boundaries, bulk_tad_d):
		self.shared_boundaries = list(np.random.choice(np.arange(self.shape), self.K, replace=False)) + list(bulk_tad_d)
		self.shared_boundaries = list(np.unique(self.shared_boundaries))
		self.shared_boundaries.sort()
		self.shared_boundaries = np.random.choice(self.shared_boundaries, self.K, replace=False)
		self.shared_boundaries = list(np.unique(self.shared_boundaries))
		self.shared_boundaries.sort()
		
		# Too close check:
		shared_boundaries = np.array(self.shared_boundaries)
		dis = shared_boundaries[1:] - shared_boundaries[:-1]
		shared_boundaries = [shared_boundaries[0]] + list(shared_boundaries[1:][dis > 1])
		self.shared_boundaries = list(shared_boundaries)
		self.shared_boundaries.sort()
		
		print("initialized boundaries", self.shared_boundaries)
		
		cum_score = np.cumsum(sc_score, axis=-1)
		n_cell = sc_score.shape[0]
		nochange_count = 0
		
		sc_assignment = [[] for b in sc_boundaries]
		
		epoch_count = 0
		
		while True:
			# Correspondence assignment
			search_range = []
			# print ("start assigning states")
			bar = trange(n_cell, desc="%s Epoch %d E-step: - " % (self.print_identifier, epoch_count))
			for cell in bar:
				c_bound = sc_boundaries[cell]
				
				cell, sc_assignment_cell, search_range_cell = self.assign( cell, len(self.shared_boundaries), self.shared_boundaries, sc_score[cell], cum_score[
					cell], c_bound)
				sc_assignment[cell] = sc_assignment_cell
				search_range.append(search_range_cell)
			
			
			
			
			search_range = np.stack(search_range, axis=-1)
			search_range_left = np.min(search_range[:, 0, :], axis=-1)
			search_range_right = np.max(search_range[:, 1, :], axis=-1)
			search_range = np.stack([search_range_left, search_range_right], axis=-1)
			
			change = 0
			pool = ProcessPoolExecutor(max_workers=5)
			p_list = []
			# Update center
			bar =  trange(len(self.shared_boundaries), desc="%s Epoch %d M-step: - " % (self.print_identifier, epoch_count))
			for j in range(len(shared_boundaries)):
				start, end = search_range[j]
				if end > start:
					p_list.append(pool.submit(self.update, n_cell, sc_boundaries, sc_assignment, sc_score, cum_score, j, start, end))
					# updated, j = self.update(n_cell, sc_boundaries, sc_assignment, sc_score, cum_score, j, start, end)
			for p in as_completed(p_list):
				bar.update(1)
				updated, j = p.result()
					
				if self.shared_boundaries[j] != updated:
					change += 1
				self.shared_boundaries[j] = updated
			pool.shutdown(wait=True)
			if change == 0:
				nochange_count += 1
			else:
				nochange_count = 0
			if nochange_count >= 2:
				break
			epoch_count += 1
			
			bar.set_description("%s Epoch: %d - update_ratio = %f " % (self.print_identifier, epoch_count, change / len(self.shared_boundaries)),
			                    refresh=False)
			self.shared_boundaries = list(np.unique(self.shared_boundaries))
			self.shared_boundaries.sort()
		
		
		# Too close check:
		shared_boundaries = np.array(self.shared_boundaries)
		dis = shared_boundaries[1:] - shared_boundaries[:-1]
		too_close = np.where(dis <= 1)[0]
		cat_assignment = np.concatenate(sc_assignment, axis=0)
		unique_assignment, count = np.unique(cat_assignment, return_counts=True)
		assign_count = {}
		for u,c in zip(unique_assignment, count):
			assign_count[u] = c
		# kept_list = list(range(len(shared_boundaries)))
		if len(too_close) > 0:
			# print ("find too close", too_close)
			for c in too_close:
				if assign_count[c] > assign_count[c+1]:
					try:
						# kept_list.remove(c + 1)
						shared_boundaries[c + 1] = shared_boundaries[c]
					except:
						pass
				else:
					try:
						# kept_list.remove(c)
						shared_boundaries[c] = shared_boundaries[c + 1]
					except:
						pass
		# shared_boundaries = shared_boundaries[np.array(kept_list)]
		self.shared_boundaries = np.array(self.shared_boundaries)
		calibrated_sc_boundaries = []
		for assignment in sc_assignment:
			# print (assignment)
			calibrated_sc_boundaries.append(np.unique(self.shared_boundaries[np.array(assignment).astype('int')]))
		
		calibrated_sc_boundaries = np.array(calibrated_sc_boundaries)
		self.shared_boundaries = np.unique(self.shared_boundaries)
		boundaries2assignment = {k:v for (v,k) in enumerate(self.shared_boundaries)}
		new_sc_assignment = np.array([[boundaries2assignment[b] for b in calib_b] for calib_b in calibrated_sc_boundaries])
		
		print (self.print_identifier, "finished")
		return self.shared_boundaries, new_sc_assignment, calibrated_sc_boundaries