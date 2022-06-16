import multiprocessing as mp
import time
import warnings
import torch.optim
from Higashi_backend.Modules import *
from Higashi_backend.Functions import *
from Higashi_backend.utils import *
from Impute import impute_process
import argparse
import resource
from scipy.sparse import csr_matrix
from scipy.sparse.csr import get_csr_submatrix
from sklearn.preprocessing import StandardScaler
import pickle
import subprocess
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
torch.backends.cudnn.benchmark = True
torch.set_default_dtype(torch.float32)

def parse_args():
	parser = argparse.ArgumentParser(description="Higashi main program")
	parser.add_argument('-c', '--config', type=str, default="../config_dir/config_ramani.JSON")
	parser.add_argument('-s', '--start', type=int, default=1)
	
	return parser.parse_args()


def get_free_gpu(num=1):
	os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > ./tmp')
	memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
	if len(memory_available) > 0:
		max_mem = np.max(memory_available)
		if num == 1:
			ids = np.where(memory_available == max_mem)[0]
			chosen_id = int(np.random.choice(ids, 1)[0])
			print("setting to gpu:%d" % chosen_id)
			torch.cuda.set_device(chosen_id)
			return "cuda:%d" % chosen_id
		else:
			ids = np.argsort(memory_available)[::-1][:num]
			return ids
		
	else:
		return


def forward_batch_hyperedge(model, loss_func, batch_data, batch_weight, batch_chrom, batch_to_neighs, y, chroms_in_batch):
	x = batch_data
	w = batch_weight
	# plus one, because chr1 - id 0 - NN1
	pred, pred_var, pred_proba = model(x, (batch_chrom, batch_to_neighs), chroms_in_batch=chroms_in_batch+1)
	
	if use_recon:
		adj = node_embedding_init.embeddings[0](cell_ids).float()
		targets = node_embedding_init.targets[0](cell_ids).float()
		embed, recon = node_embedding_init.wstack[0](adj, return_recon=True)
		mse_loss = F.mse_loss(recon, targets)
	else:
		mse_loss = torch.as_tensor([0], dtype=torch.float).to(device, non_blocking=True)
		
		
	if mode == 'classification':
		main_loss = loss_func(pred, y, weight=w)
		
	elif mode == 'rank':
		pred = F.softplus(pred)
		diff = (pred.view(-1, 1) - pred.view(1, -1)).view(-1)
		diff_w = (w.view(-1, 1) - w.view(1, -1)).view(-1)
		mask_rank = torch.abs(diff_w) > rank_thres
		diff = diff[mask_rank].float()
		diff_w = diff_w[mask_rank]
		label = (diff_w > 0).float()
		main_loss = loss_func(diff, label)
		
		if not use_recon:
			if neg_num > 0:
				mask_w_eq_zero = w == 0
				makeitzero = F.mse_loss(w[mask_w_eq_zero].float(), pred[mask_w_eq_zero].float())
				mse_loss += makeitzero

	elif mode == 'zinb':
		pred = F.softplus(pred)
		extra = (cell_feats1[batch_data[:, 0]-1]).view(-1, 1)
		pred = pred * extra
		pred_var = F.softplus(pred_var)

		pred = torch.clamp(pred, min=1e-8, max=1e8)
		pred_var = torch.clamp(pred_var, min=1e-8, max=1e8)
		main_loss = -log_zinb_positive(w.float(), pred.float(), pred_var.float(), pred_proba.float())
		main_loss = main_loss.mean()
	elif mode == 'regression':
		pred = pred.float().view(-1)
		w = w.float().view(-1)
		main_loss =  F.mse_loss(pred, w)
	else:
		print ("wrong mode")
		raise EOFError
	
	return pred, main_loss, mse_loss


def train_epoch(model, loss_func, training_data_generator, optimizer_list, train_pool, train_p_list):
	loss_func = loss_func
	
	model.train()
	
	bce_total_loss = 0
	mse_total_loss = 0
	final_batch_num = 0
	
	batch_num = int(update_num_per_training_epoch / collect_num)
	
	y_list, w_list, pred_list = [], [], []
	
	bar = trange(batch_num * collect_num, desc=' - (Training) ', leave=False, )
	finish_count = 0
	
	def ps1(batch_edge_big, batch_y_big, batch_edge_weight_big, batch_chrom_big, batch_to_neighs_big, chroms_in_batch):
		batch_edge_big = np2tensor_hyper(batch_edge_big, dtype=torch.long)
		batch_y_big, batch_edge_weight_big = torch.from_numpy(batch_y_big), torch.from_numpy(batch_edge_weight_big)
		
		batch_edge_big, batch_y_big, batch_edge_weight_big = batch_edge_big.to(device,
		                                                                       non_blocking=True), batch_y_big.to(
			device, non_blocking=True), batch_edge_weight_big.to(device, non_blocking=True)
		size = int(len(batch_edge_big) / collect_num)
		for j in range(collect_num):
			batch_edge, batch_edge_weight, batch_y, batch_chrom, batch_to_neighs = batch_edge_big[
			                                                                       j * size: min((j + 1) * size,
			                                                                                     len(batch_edge_big))], \
			                                                                       batch_edge_weight_big[
			                                                                       j * size: min((j + 1) * size,
			                                                                                     len(batch_edge_big))], \
			                                                                       batch_y_big[
			                                                                       j * size: min((j + 1) * size,
			                                                                                     len(batch_edge_big))], \
			                                                                       batch_chrom_big[
			                                                                       j * size: min((j + 1) * size,
			                                                                                     len(batch_edge_big))], \
			                                                                       batch_to_neighs_big[j]
			
			pred, loss_bce, loss_mse = forward_batch_hyperedge(model, loss_func, batch_edge,
			                                                   batch_edge_weight, batch_chrom,
			                                                   batch_to_neighs, y=batch_y,
			                                                   chroms_in_batch=chroms_in_batch)
			
			y_list.append(batch_y.detach().cpu())
			w_list.append(batch_edge_weight.detach().cpu())
			pred_list.append(pred.detach().cpu())
			
			if use_recon:
				for opt in optimizer_list:
					opt.zero_grad(set_to_none=True)
				loss_bce.backward(retain_graph=True)
				try:
					main_norm = node_embedding_init.wstack[0].weight_list[0].grad.data.norm(2)
				except:
					main_norm = 0.0
				
				for opt in optimizer_list:
					opt.zero_grad(set_to_none=True)
				loss_mse.backward(retain_graph=True)
				
				recon_norm = node_embedding_init.wstack[0].weight_list[0].grad.data.norm(2)
				ratio = beta * main_norm / recon_norm
				ratio1 = max(ratio, 100 * median_total_sparsity_cell - 3)
				
				if contractive_flag:
					contractive_loss = 0.0
					for i in range(len(node_embedding_init.wstack[0].weight_list)):
						contractive_loss += torch.sum(node_embedding_init.wstack[0].weight_list[i] ** 2)
						contractive_loss += torch.sum(node_embedding_init.wstack[0].reverse_weight_list[i] ** 2)
				
				else:
					contractive_loss = 0.0
			
			else:
				contractive_loss = 0.0
				ratio = 0.0
				ratio1 = 0.0
			
			train_loss = alpha * loss_bce + ratio1 * loss_mse + contractive_loss_weight * contractive_loss
			for opt in optimizer_list:
				opt.zero_grad(set_to_none=True)
			# backward
			train_loss.backward()
			
			# update parameters
			for opt in optimizer_list:
				opt.step()
			
			bar.update(n=1)
			bar.set_description(" - (Training) BCE:  %.3f MSE: %.3f Loss: %.3f norm_ratio: %.2f" %
			                    (loss_bce.item(), loss_mse.item(), train_loss.item(), ratio1),
			                    refresh=False)
		
		if train_p_list is not None:
			train_p_list.remove(p)
			
			while len(train_p_list) < batch_num:
				edges_part, edges_chrom, edge_weight_part, chroms_in_batch = training_data_generator.next_iter()
				train_p_list.append(
					train_pool.submit(one_thread_generate_neg, edges_part, edges_chrom, edge_weight_part, collect_num, True,
					                  chroms_in_batch))
		
		return loss_bce.item(), loss_mse.item()
	
	
	
	if train_p_list is not None:
		while len(train_p_list) < batch_num:
			edges_part, edges_chrom, edge_weight_part, chroms_in_batch = training_data_generator.next_iter()
			train_p_list.append(train_pool.submit(one_thread_generate_neg, edges_part, edges_chrom, edge_weight_part, collect_num, True,
			                          chroms_in_batch))
		for p in as_completed(train_p_list):
			batch_edge_big, batch_y_big, batch_edge_weight_big, batch_chrom_big, batch_to_neighs_big, chroms_in_batch = p.result()
			loss_bce, loss_mse = ps1(batch_edge_big, batch_y_big, batch_edge_weight_big, batch_chrom_big, batch_to_neighs_big, chroms_in_batch)
			bce_total_loss += loss_bce
			mse_total_loss += loss_mse
			final_batch_num += 1
			finish_count += 1
			if finish_count == batch_num:
				break
	else:
		for i in range(batch_num):
			edges_part, edges_chrom, edge_weight_part, chroms_in_batch = training_data_generator.next_iter()
			batch_edge_big, batch_y_big, batch_edge_weight_big, batch_chrom_big, \
			batch_to_neighs_big, chroms_in_batch = one_thread_generate_neg(edges_part, edges_chrom, edge_weight_part, collect_num, True,
			                          chroms_in_batch)
			loss_bce, loss_mse = ps1(batch_edge_big, batch_y_big, batch_edge_weight_big, batch_chrom_big,
			                         batch_to_neighs_big, chroms_in_batch)
			bce_total_loss += loss_bce
			mse_total_loss += loss_mse
			final_batch_num += 1
			finish_count += 1
			if finish_count == batch_num:
				break
			

	
	
	y = torch.cat(y_list)
	w = torch.cat(w_list)
	pred = torch.cat(pred_list)
	
	auc1, auc2, str1, str2 = roc_auc_cuda(w, pred)
	bar.close()
	return bce_total_loss / final_batch_num, mse_total_loss / final_batch_num, accuracy(
		y.view(-1), pred.view(-1)), auc1, auc2, str1, str2, train_pool, train_p_list


def eval_epoch(model, loss_func, validation_data_generator, p_list=None, eval_pool=None):
	"""Epoch operation in evaluation phase"""
	bce_total_loss = 0
	model.eval()
	with torch.no_grad():
		y_list, w_list, pred_list = [], [], []
		
		bar = tqdm(range(update_num_per_eval_epoch), desc='  - (Validation)   ', leave=False)
		if p_list is None and eval_pool is None:
			for i in range(update_num_per_eval_epoch):
				edges_part, edges_chrom, edge_weight_part, chroms_in_batch = validation_data_generator.next_iter()
				batch_x, batch_y, batch_w, batch_chrom, \
				batch_to_neighs, chroms_in_batch = one_thread_generate_neg(edges_part, edges_chrom,
				                                                           edge_weight_part, 1,
				                                                           False, chroms_in_batch)
				batch_x = np2tensor_hyper(batch_x, dtype=torch.long)
				batch_y, batch_w = torch.from_numpy(batch_y), torch.from_numpy(batch_w)
				batch_x, batch_y, batch_w = batch_x.to(device, non_blocking=True), batch_y.to(device,
				                                                                              non_blocking=True), batch_w.to(
					device, non_blocking=True)
				
				pred_batch, eval_loss, eval_mse = forward_batch_hyperedge(model, loss_func, batch_x, batch_w,
				                                                          batch_chrom, batch_to_neighs[0], y=batch_y,
				                                                          chroms_in_batch=chroms_in_batch)
				
				bce_total_loss += eval_loss.item()
				
				y_list.append(batch_y.detach().cpu())
				w_list.append(batch_w.detach().cpu())
				pred_list.append(pred_batch.detach().cpu())
				
				bar.update(n=1)
				bar.set_description(" - (Validation) BCE:  %.3f MSE: %.3f " %
				                    (eval_loss.item(), eval_mse.item()),
				                    refresh=False)
		else:
			if p_list is None:
				p_list = []
				for i in range(update_num_per_eval_epoch):
					edges_part, edges_chrom, edge_weight_part, chroms_in_batch = validation_data_generator.next_iter()
					p_list.append(
						eval_pool.submit(one_thread_generate_neg, edges_part, edges_chrom, edge_weight_part, 1, False, chroms_in_batch))
			
			for p in as_completed(p_list):
				batch_x, batch_y, batch_w, batch_chrom, batch_to_neighs, chroms_in_batch = p.result()
				batch_x = np2tensor_hyper(batch_x, dtype=torch.long)
				batch_y, batch_w = torch.from_numpy(batch_y), torch.from_numpy(batch_w)
				batch_x, batch_y, batch_w = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True), batch_w.to(device, non_blocking=True)
				
				pred_batch, eval_loss, eval_mse = forward_batch_hyperedge(model, loss_func, batch_x, batch_w, batch_chrom, batch_to_neighs[0], y=batch_y, chroms_in_batch=chroms_in_batch)
				
				
				bce_total_loss += eval_loss.item()
				
				y_list.append(batch_y.detach().cpu())
				w_list.append(batch_w.detach().cpu())
				pred_list.append(pred_batch.detach().cpu())
				
				
				bar.update(n=1)
				bar.set_description(" - (Validation) BCE:  %.3f MSE: %.3f " %
									(eval_loss.item(), eval_mse.item()),
									refresh=False)
			
		bar.close()
		
		y = torch.cat(y_list)
		w = torch.cat(w_list)
		pred = torch.cat(pred_list)
		
		auc1, auc2, str1, str2 = roc_auc_cuda(w, pred)
		
	return bce_total_loss / (update_num_per_eval_epoch), accuracy(y.view(-1), pred.view(-1)), auc1, auc2, str1, str2


def check_nonzero(x, c):
	# minus 1 because add padding index
	mtx = sparse_chrom_list_dict[c][x[0]-1]
	M, N = mtx.shape
	if mem_efficient_flag:
		row_start = max(0, x[1]-1-num_list[c]-1)
		row_end = min(x[1]-1-num_list[c]+2, N-1)
		col_start = max(0, x[2]-1-num_list[c]-1)
		col_end = min(x[2]-1-num_list[c]+2, N-1)
	else:
		row_start = x[1]-1-num_list[c]
		row_end = min(row_start + 1, N-1)
		col_start = x[2]-1-num_list[c]
		col_end = min(col_start + 1, N-1)
	try:
		indptr, nbrs, nbr_value = get_csr_submatrix(
			M, N, mtx.indptr, mtx.indices, mtx.data, row_start, row_end, col_start, col_end)
	except:
		print (M, N, row_start, row_end, col_start, col_end, mem_efficient_flag)
	a =  len(nbr_value) > 0
	
	return a

def check_nonzero_old(x, c):
	# minus 1 because add padding index
	if mem_efficient_flag:
		dim = sparse_chrom_list_dict[c][x[0]-1].shape[-1]
		a = sparse_chrom_list_dict[c][x[0]-1][max(0, x[1]-1-num_list[c]-1):min(x[1]-1-num_list[c]+2, dim-1),
		       max(0, x[2]-1-num_list[c]-1):min(x[2]-1-num_list[c]+2, dim-1)].sum() > 0
	else:
		a =  sparse_chrom_list_dict[c][x[0]-1][x[1]-1-num_list[c], x[2]-1-num_list[c]] > 0
	return a

def generate_negative_cpu(x, x_chrom, forward=True):
	global pair_ratio
	rg = np.random.default_rng()
	neg_list, neg_chrom = np.zeros((x.shape[0] * neg_num, x.shape[1]), dtype='int'), \
	                                 np.zeros((x.shape[0] * neg_num), dtype='int8')
	
	if forward:
		func1 = pass_
	else:
		func1 = tqdm
	
	success_count = 0
	
	change_list_all = rg.integers(0, x.shape[-1], (len(x), neg_num))
	simple_or_hard_all = rg.random((len(x), neg_num))
	for j, sample in enumerate(func1(x)):
		
		for i in range(neg_num):
			temp = np.copy(sample)
			change = change_list_all[j, i]
			trial = 0
			while check_nonzero(temp, x_chrom[j]):
			# while flag:
				if steps == 1:
					temp = np.copy(sample)
				
				# Try too many times on one sample, move on
				trial += 1
				if trial >= 50:
					temp = ""
					break
				
				# hard mode, Only change one node
				if simple_or_hard_all[j, i] <= pair_ratio:
					start, end = start_end_dict[int(temp[change])]
					
					# It's changing the bin
					if change != 0:
						other_bin = 2 if change == 1 else 2
						other_bin = temp[other_bin]
						start = max(start, other_bin - max_bin)
						end = min(end, other_bin + max_bin)
						
						temp[change] = np.random.randint(
							int(start), int(end), 1) + 1
					else:
						temp[change] = rg.choice(end - start) + start + 1
					
				else:
					start, end = start_end_dict[int(temp[0])]
						
					temp[0] = rg.choice(end - start) + start + 1
					
					start, end = start_end_dict[int(temp[1])]
					
					temp[1] = rg.choice(end - start) + start + 1
					
					start = max(start, temp[1] - max_bin)
					end = min(end, temp[1] + max_bin)
					
					
					temp[2] = rg.choice(end - start) + start + 1
					
				
				temp.sort()
				
				# Not a suitable sample
				if ((temp[2] - temp[1]) >= max_bin) or ((temp[2] - temp[1]) <= 1):
					temp = np.copy(sample)
			
			if len(temp) > 0:
				neg_list[success_count, :] = temp
				neg_chrom[success_count] = x_chrom[j]
				success_count += 1
			if success_count == neg_num * len(x):
				break
	
	return neg_list[:success_count], neg_chrom[:success_count]


def to_neighs_to_mask(to_neighs):
	samp_neighs = to_neighs.reshape((-1))
	unique_nodes = {}
	unique_nodes_list = []
	
	count = 0
	column_indices = []
	row_indices = []
	if torch_has_csr:
		crow_indices = [0]
	else:
		row_indices = []
	v = []
	
	for i, samp_neigh in enumerate(samp_neighs):
		if len(samp_neigh) == 0:
			continue
		
		w = samp_neigh[1]
		samp_neigh = samp_neigh[0]

		w /= np.sum(w)
		
		try:
			for n in samp_neigh:
				
				if n not in unique_nodes:
					unique_nodes[n] = count
					unique_nodes_list.append(n)
					count += 1
				column_indices.append(unique_nodes[n])

				if not torch_has_csr:
					row_indices.append(i)
			if torch_has_csr:
				crow_indices.append(crow_indices[-1] + len(samp_neigh))
		except:
			print(i, samp_neigh, samp_neighs[i])
			raise EOFError
		
		v.append(w)
	
	
	v = np.concatenate(v, axis=0)
	
	unique_nodes_list = torch.LongTensor(unique_nodes_list)
	
	if torch_has_csr:
		return (torch.from_numpy(np.asarray([crow_indices, column_indices])),  torch.from_numpy(v), unique_nodes_list)
	else:
		return (torch.from_numpy(np.asarray([row_indices, column_indices])), torch.from_numpy(v), unique_nodes_list)


def sum_duplicates(col, data):
	order = np.argsort(col)
	col = col[order]
	data = data[order]
	unique_mask = (col[1:] != col[:-1])
	unique_mask = np.append(True, unique_mask)
	col = col[unique_mask]
	unique_inds, = np.nonzero(unique_mask)
	data = np.add.reduceat(data, unique_inds)
	return col, data

def one_thread_generate_neg(edges_part, edges_chrom, edge_weight, collect_num=1, training=False, chroms_in_batch=None):
	if neg_num == 0:
		y = np.ones((len(edges_part), 1))
		w = np.ones((len(edges_part), 1)) * edge_weight.reshape((-1, 1))
		x = edges_part
	else:
		try:
			neg_list, neg_chrom = generate_negative_cpu(edges_part, edges_chrom, True)
			neg_list = np.array(neg_list)[: len(edges_part) * neg_num, :]
			neg_chrom = np.array(neg_chrom)[: len(edges_part) * neg_num]
			if len(neg_list) == 0:
				raise EOFError
			
			correction = 1.0 if mode == "classification" else 0.0
			y = np.concatenate([np.ones((len(edges_part), 1)),
			                    np.zeros((len(neg_list), 1))])
			w = np.concatenate([np.ones((len(edges_part), 1)) * edge_weight.reshape((-1, 1)),
								np.ones((len(neg_list), 1)) * correction])
			x = np.concatenate([edges_part, neg_list])
			x_chrom = np.concatenate([edges_chrom, neg_chrom])
		except Exception as e:
			print("error from generate neg", e)
			raise EOFError
	
	new_x, new_y, new_w, new_x_chrom = [], [], [], []
		
	
	if isinstance(higashi_model.encode1.dynamic_nn, GraphSageEncoder_with_weights):
		cell_ids = np.stack([x[:, 0], x[:, 0]], axis=-1).reshape((-1))
		bin_ids = x[:, 1:].reshape((-1))
		remove_bin_ids = np.copy(x[:, 1:])
		remove_bin_ids = remove_bin_ids[:, ::-1]
		nodes_chrom = np.stack([x_chrom, x_chrom], axis=-1).reshape((-1))
		to_neighs = []
		
		rg = np.random.default_rng()
		remove_or_not = rg.random(len(nodes_chrom))
		
		for i, (c, cell_id, bin_id, remove_bin_id) in enumerate(zip(nodes_chrom, cell_ids, bin_ids, remove_bin_ids.reshape((-1)))):
			if precompute_weighted_nbr:
				mtx = sparse_chrom_list_GCN[c][cell_id - 1]
				row = bin_id - 1 - num_list[c]
				M, N = mtx.shape
				indptr, nbrs, nbr_value = get_csr_submatrix(
					M, N, mtx.indptr, mtx.indices, mtx.data, row, row + 1, 0, N)
			else:
				if weighted_adj:
					# row = 0
					# for nbr_cell in cell_neighbor_list[cell_id]:
					# 	balance_weight = weight_dict[(nbr_cell, cell_id)]
					# 	temp = balance_weight * sparse_chrom_list[c][nbr_cell - 1][bin_id - 1 - num_list[c]]
					#
					# 	# if training and (nbr_cell == cell_id) and (remove_or_not[i] >= 0.6):
					# 	# 	if len(temp.data) > 1:
					# 	# 		temp[0, remove_bin_id - 1 - num_list[c]] = 0
					# 	row = row + temp
					# M, N = row.shape
					# indptr, nbrs, nbr_value = get_csr_submatrix(
					# 	M, N, row.indptr, row.indices, row.data, 0, 1, 0, N)
					
					row_record, row_weight_record = [], []
					for nbr_cell in cell_neighbor_list[cell_id]:
						balance_weight = weight_dict[(nbr_cell, cell_id)]
						mtx = sparse_chrom_list_GCN[c][nbr_cell - 1]
						row = bin_id - 1 - num_list[c]
						M, N = mtx.shape
						indptr, nbrs_pt, nbr_value_pt = get_csr_submatrix(
							M, N, mtx.indptr, mtx.indices, mtx.data, row, row + 1, 0, N)
						row_record.append(nbrs_pt)
						row_weight_record.append(nbr_value_pt * balance_weight)
					nbrs, nbr_value = sum_duplicates(np.concatenate(row_record), np.concatenate(row_weight_record))
				else:
					mtx = sparse_chrom_list_GCN[c][cell_id - 1]
					row = bin_id - 1 - num_list[c]
					M, N = mtx.shape
					indptr, nbrs, nbr_value = get_csr_submatrix(
						M, N, mtx.indptr, mtx.indices, mtx.data, row, row + 1, 0, N)
					
					
			if training and (remove_or_not[i] >= 0.6):
				if len(nbrs) > 1:
					mask = nbrs != (remove_bin_id - 1 - num_list[c])
					nbrs = nbrs[mask]
					nbr_value = nbr_value[mask]
			
			nbr_value = np.log1p(nbr_value)
			nbrs = nbrs.reshape((-1)) + 1 + num_list[c]

			if type(nbrs) is not np.ndarray:
				print (row, nbrs)
			if len(nbrs) > 0:
				temp = [nbrs, nbr_value]
			else:
				temp = []
			to_neighs.append(temp)

		# Force to append an empty list and remove it, such that np.array won't broadcasting shapes
		to_neighs.append([])
		to_neighs = np.array(to_neighs)[:-1]
		to_neighs = np.array(to_neighs, dtype='object').reshape((len(x), 2))
		
		size = int(len(x) / collect_num)
		to_neighs_new = []
		
		if collect_num == 1:
			index = np.random.permutation(len(x))
			x, y, w, x_chrom, remove_bin_ids = x[index], \
			                   y[index], \
			                   w[index], \
			                   x_chrom[index], \
							   remove_bin_ids[index]
			to_neighs = [to_neighs_to_mask(to_neighs[index])]
		else:

			for j in range(collect_num):
				x_part, y_part, w_part, x_chrom_part, to_neighs_part, remove_bin_ids_part = x[
														 j * size: min((j + 1) * size, len(x))], y[
														 j * size: min((j + 1) * size, len(x))], w[
														 j * size: min((j + 1) * size, len(x))], x_chrom[
														 j * size: min((j + 1) * size, len(x))], to_neighs[
														 j * size: min((j + 1) * size, len(x))], remove_bin_ids[
														 j * size: min((j + 1) * size, len(x))]
	
				index = np.random.permutation(len(x_part))

				x_part, y_part, w_part, x_chrom_part, to_neighs_part, remove_bin_ids_part = x_part[index], \
																							y_part[index],\
																							w_part[index],\
																							x_chrom_part[index],\
																							to_neighs_part[index],\
																							remove_bin_ids_part[index]
				new_x.append(x_part)
				new_y.append(y_part)
				new_w.append(w_part)
				new_x_chrom.append(x_chrom_part)
				to_neighs_new.append(to_neighs_to_mask(to_neighs_part))
			to_neighs = to_neighs_new
	else:
		size = int(len(x) / collect_num)
		if collect_num == 1:
			index = np.random.permutation(len(x))
			x, y, w, x_chrom = x[index], \
                               y[index], \
                               w[index], \
                               x_chrom[index]
			to_neighs = [[]]
		else:
			for j in range(collect_num):
	
				x_part, y_part, w_part, x_chrom_part = x[
														 j * size: min((j + 1) * size, len(x))], y[
														 j * size: min((j + 1) * size, len(x))], w[
														 j * size: min((j + 1) * size, len(x))], x_chrom[
														 j * size: min((j + 1) * size, len(x))]
	
				index = np.random.permutation(len(x_part))
				x_part, y_part, w_part, x_chrom_part = x_part[index], \
														y_part[index],\
														w_part[index],\
														x_chrom_part[index]
				new_x.append(x_part)
				new_y.append(y_part)
				new_w.append(w_part)
				new_x_chrom.append(x_chrom_part)
			to_neighs = [[] for i in range(collect_num)]
	
			x = np.concatenate(new_x, axis=0)
			y = np.concatenate(new_y, axis=0)
			w = np.concatenate(new_w, axis=0)
			x_chrom = np.concatenate(new_x_chrom, axis=0)
	return x, y, w, x_chrom, to_neighs, chroms_in_batch


def train(model, loss, training_data_generator, validation_data_generator, optimizer, epochs, load_first, save_embed=False, save_name=""):
	global pair_ratio
	no_improve = 0
	if load_first:
		checkpoint = torch.load(save_path+save_name)
		model.load_state_dict(checkpoint['model_link'])

	best_train_loss = 1000
	
	if save_embed:
		save_embeddings(model)
	
	
	if cpu_num > 1:
		train_pool = ProcessPoolExecutor(max_workers=cpu_num)
		eval_pool = ProcessPoolExecutor(max_workers=cpu_num)
		train_p_list = []
	else:
		print ("no parallel")
		train_pool, train_p_list = None, None
		eval_pool = None
	
	for epoch_i in range(epochs):
		if save_embed:
			save_embeddings(model)
		
		print('[ Epoch', epoch_i, 'of', epochs, ']')
		if eval_pool is not None:
			eval_p_list = []
			for i in range(update_num_per_eval_epoch):
				edges_part, edges_chrom, edge_weight_part, chroms_in_batch = validation_data_generator.next_iter()
				eval_p_list.append(
					eval_pool.submit(one_thread_generate_neg, edges_part, edges_chrom, edge_weight_part, 1, False, chroms_in_batch))
		else:
			eval_p_list = None
		
		start = time.time()
		
		bce_loss, mse_loss, train_accu, auc1, auc2, str1, str2, train_pool, train_p_list = train_epoch(
			model, loss, training_data_generator, optimizer, train_pool, train_p_list)
		print('  - (Training)   bce: {bce_loss: 7.4f}, mse: {mse_loss: 7.4f}, '
			  ' acc: {accu:3.3f} %, {str1}: {auc1:3.3f}, {str2}: {auc2:3.3f}, '
			  'elapse: {elapse:3.3f} s'.format(
			bce_loss=bce_loss,
			mse_loss=mse_loss,
			accu=100 *
				 train_accu,
			str1=str1,
			auc1=auc1,
			str2=str2,
			auc2=auc2,
			elapse=(time.time() - start)))
		
			
		start = time.time()
		valid_bce_loss, valid_accu, valid_auc1, valid_auc2, str1, str2 = eval_epoch(
			model, loss, validation_data_generator, eval_p_list, eval_pool)
		print('  - (Validation-hyper) bce: {bce_loss: 7.4f},'
			  '  acc: {accu:3.3f} %,'
			  '{str1}: {auc1:3.3f}, {str2}: {auc2:3.3f},'
			  'elapse: {elapse:3.3f} s'.format(
			bce_loss=valid_bce_loss,
			accu=100 *
				 valid_accu,
			str1=str1,
			auc1=valid_auc1,
			auc2=valid_auc2,
			str2=str2,
			elapse=(time.time() - start)))
		
		# Dynamic pair ratio for stage one
		if dynamic_pair_ratio:
			if pair_ratio < 0.5:
				pair_ratio += 0.1
			else:
				pair_ratio = 0.5
				# scheduler.step(bce_loss)
			print("pair_ratio", pair_ratio)
		
		if not dynamic_pair_ratio:
			if bce_loss < best_train_loss - 1e-3:
				best_train_loss = bce_loss
				no_improve = 0
			else:
				no_improve += 1
				
			# scheduler.step(bce_loss)
		
		if no_improve >= 5:
			print ("no improvement early stopping")
			break
		print ("no improve", no_improve)

		if epoch_i % 5 == 0:
			checkpoint = {
				'model_link': model.state_dict(),
				'epoch': epoch_i}
			torch.save(checkpoint, save_path + save_name)


	start = time.time()
	
	valid_bce_loss, valid_accu, valid_auc1, valid_auc2, _, _ = eval_epoch(model, loss, validation_data_generator, None, eval_pool)
	print('  - (Validation-hyper) bce: {bce_loss: 7.4f},'
		  '  acc: {accu:3.3f} %,'
		  ' auc: {auc1:3.3f}, aupr: {auc2:3.3f},'
		  'elapse: {elapse:3.3f} s'.format(
		bce_loss=valid_bce_loss,
		accu=100 *
			 valid_accu,
		auc1=valid_auc1,
		auc2=valid_auc2,
		elapse=(time.time() - start)))


def save_embeddings(model):
	model.eval()
	with torch.no_grad():
		ids = torch.arange(1, num_list[-1] + 1).long().to(device, non_blocking=True).view(-1)
		embeddings = []
		for j in range(math.ceil(len(ids) / batch_size)):
			x = ids[j * batch_size:min((j + 1) * batch_size, len(ids))]
			
			embed = node_embedding_init(x)
			embed = embed.detach().cpu().numpy()
			embeddings.append(embed)
		
		embeddings = np.concatenate(embeddings, axis=0)
		for i in range(len(num_list)):
			start = 0 if i == 0 else num_list[i - 1]
			static = embeddings[int(start):int(num_list[i])]
			
			
			if i == 0:
				try:
					old_static = np.load(os.path.join(embed_dir, "%s_%d_origin.npy" % (embedding_name, i)))
					update_rate = np.sum((old_static - static) ** 2, axis=-1) / np.sum(old_static ** 2, axis=-1)
					print("update_rate: %f\t%f" % (np.min(update_rate), np.max(update_rate)))
				except Exception as e:
					pass
				
			np.save(os.path.join(embed_dir, "%s_%d_origin.npy" % (embedding_name, i)), static)
	
	torch.cuda.empty_cache()
	return embeddings

		
def generate_attributes():
	embeddings = []
	targets = []
	cell_node_feats = []

	with h5py.File(os.path.join(temp_dir, "node_feats.hdf5"), "r") as save_file:

		for c in chrom_list:
			a = np.array(save_file["cell"]["%d" % chrom_list.index(c)])
			cell_node_feats.append(a)

		if coassay:
			print ("coassay")
			cell_attributes = np.load(os.path.join(temp_dir, "pretrain_coassay.npy")).astype('float32')
			targets.append(cell_attributes)
			cell_node_feats = StandardScaler().fit_transform(cell_node_feats)
			embeddings.append(cell_node_feats)
		else:
			if num[0] >= 10:
				cell_node_feats1 = remove_BE_linear(cell_node_feats, config, data_dir, cell_feats1)
				cell_node_feats1 = StandardScaler().fit_transform(cell_node_feats1.reshape((-1, 1))).reshape((len(cell_node_feats1), -1))
				cell_node_feats2 = [StandardScaler().fit_transform(x) for x in cell_node_feats]
				cell_node_feats2 = remove_BE_linear(cell_node_feats2, config, data_dir, cell_feats1)
				cell_node_feats2 = StandardScaler().fit_transform(cell_node_feats2.reshape((-1, 1))).reshape((len(cell_node_feats2), -1))
			else:
				cell_node_feats1 = remove_BE_linear(cell_node_feats, config, data_dir, cell_feats1)
				cell_node_feats2 = cell_node_feats1
				
			targets.append(cell_node_feats2.astype('float32'))
			embeddings.append(cell_node_feats1.astype('float32'))


		for i, c in enumerate(chrom_list):
			temp = np.array(save_file["%d" % i]).astype('float32')
			# temp = StandardScaler().fit_transform(temp.reshape((-1, 1))).reshape((len(temp), -1))
			chrom = np.zeros((len(temp), len(chrom_list))).astype('float32')
			chrom[:, i] = 1
			list1 = [temp, chrom]

			temp = np.concatenate(list1, axis=-1)
			embeddings.append(temp)
			targets.append(temp)
	
	print("start making attribute")
	attribute_all = []
	for i in range(len(num) - 1):
		chrom = np.zeros((num[i + 1], len(chrom_list)))
		chrom[:, i] = 1
		coor = np.arange(num[i + 1]).reshape((-1, 1)).astype('float32')
		coor /= num[1]
		attribute = np.concatenate([chrom, coor], axis=-1)
		# attribute = chrom
		attribute_all.append(attribute)
	
	attribute_all = np.concatenate(attribute_all, axis=0)
	attribute_dict = np.concatenate([np.zeros((num[0] + 1, attribute_all.shape[-1])), attribute_all], axis=0).astype(
		'float32')

	return embeddings, attribute_dict, targets


def get_cell_neighbor_be(start=1):
	v = np.load(os.path.join(embed_dir, "%s_0_origin.npy" % embedding_name))
	
	
	distance = pairwise_distances(v, metric='euclidean')
	distance_sorted = np.sort(distance, axis=-1)
	distance /= np.quantile(distance_sorted[:, 1:neighbor_num].reshape((-1)), q=0.25)

	cell_neighbor_list = [[] for i in range(num[0] + 1)]
	cell_neighbor_weight_list = [[] for i in range(num[0] + 1)]
	if "batch_id" in config:
		label_info = pickle.load(open(os.path.join(data_dir, "label_info.pickle"), "rb"))
		label = np.array(label_info[config["batch_id"]])
		batches = np.unique(label)
		equal_num = int(math.ceil((neighbor_num - 1) / (len(batches) - 1 )))
		
		indexs = [np.where(label == b)[0] for b in batches]
		
		for i, d in enumerate(tqdm(distance)):
			neighbor  = []
			weight = []
			b_this = label[i]
			for j in range(len(batches)):
				if b_this == batches[j]:
					neighbor.append(indexs[j][np.argsort(d[indexs[j]])][:1])
					weight.append(d[neighbor[-1]])
				else:
					neighbor.append(indexs[j][np.argsort(d[indexs[j]])][:equal_num])
					weight.append(d[neighbor[-1]])

			# print(neighbor, weight)

			weight = [w/(np.mean(w)+1e-15) for w in weight]
			neighbor = np.concatenate(neighbor)
			weight = np.concatenate(weight)

			# print(neighbor, weight)

			
			neighbor = neighbor[np.argsort(weight)]
			weight = np.sort(weight)
			
			index = np.random.permutation(len(neighbor)-1)
			neighbor[1:] = neighbor[1:][index]
			weight[1:] = weight[1:][index]
			neighbor = neighbor[:neighbor_num]
			weight = weight[:neighbor_num]
			neighbor = neighbor[np.argsort(weight)]
			weight = np.sort(weight)

		
			new_w = np.exp(-weight[start:])
			new_w /= np.sum(new_w)
			cell_neighbor_list[i + 1] = (neighbor + 1)[start:]
			cell_neighbor_weight_list[i + 1] = (new_w)
	
	return np.array(cell_neighbor_list), np.array(cell_neighbor_weight_list)


def get_cell_neighbor(start=1):
	v = np.load(os.path.join(embed_dir, "%s_0_origin.npy" % embedding_name))
	distance = pairwise_distances(v, metric='euclidean')
	distance_sorted = np.sort(distance, axis=-1)
	distance /= np.quantile(distance_sorted[:, 1:neighbor_num].reshape((-1)), q=0.25)
	cell_neighbor_list = [[] for i in range(num[0] + 1)]
	cell_neighbor_weight_list = [[] for i in range(num[0] + 1)]
	for i, d in enumerate(tqdm(distance)):
		neighbor = np.argsort(d)[:neighbor_num]
		weight = np.sort(d)[:neighbor_num]
		
		neighbor_new = neighbor
		new_w = weight
		
		new_w = np.exp(-new_w[start:])
		new_w /= np.sum(new_w)
		

		cell_neighbor_list[i + 1] = (neighbor_new + 1)[start:]
		cell_neighbor_weight_list[i + 1] = (new_w)
	
	return np.array(cell_neighbor_list), np.array(cell_neighbor_weight_list)


def mp_impute(config_path, path, name, mode, cell_start, cell_end, sparse_path, weighted_info=None, gpu_id=None):
	cmd = ["python", "Impute.py", config_path, path, name, mode, str(int(cell_start)), str(int(cell_end)), sparse_path]
	if weighted_info is not None:
		cmd += [weighted_info]
	else:
		cmd += ["None"]
		

	if gpu_id is not None:
		cmd += [str(int(gpu_id))]
	else:
		cmd += ["None"]
	print (cmd)
	subprocess.call(cmd)

	
if __name__ == '__main__':
	
	# Get parameters from config file
	args = parse_args()
	config = get_config(args.config)
	cpu_num = config['cpu_num']
	
	if cpu_num < 0:
		cpu_num = int(mp.cpu_count())
		
	if 'cpu_num_torch' in config:
		cpu_num_torch = config['cpu_num_torch']
		if cpu_num_torch < 0:
			cpu_num_torch = int(mp.cpu_count())
	else:
		cpu_num_torch = cpu_num
	print("cpu_num", cpu_num)
	gpu_num = config['gpu_num']
	print("gpu_num", gpu_num)
	
	
	if torch.cuda.is_available():
		current_device = get_free_gpu()
	else:
		current_device = 'cpu'
		torch.set_num_threads(cpu_num_torch)
		
	global pair_ratio
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print ("device", device)
	warnings.filterwarnings("ignore")
	rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
	resource.setrlimit(resource.RLIMIT_NOFILE, (3600, rlimit[1]))
	
	
	training_stage = args.start
	data_dir = config['data_dir']
	temp_dir = config['temp_dir']
	embed_dir = os.path.join(temp_dir, "embed")
	if not os.path.exists(embed_dir):
		os.mkdir(embed_dir)

	chrom_list = config['chrom_list']
	print (chrom_list)
	
	# torch_has_csr = hasattr(torch, "sparse_csr_tensor")
	# if torch_has_csr:
	# 	print ("Your pytorch has sparse_csr_matrix, will use csr instead of coo")
	# else:
	# 	print ("pytorch sparse_csr_tensor not found, fallback to coo")
	# pytorch csr in 1.10 still does not support backward
	torch_has_csr = False
	
	if gpu_num < 2:
		non_para_impute = True
		impute_pool = None
	else:
		non_para_impute = False
		impute_pool = ProcessPoolExecutor(max_workers=gpu_num)
		
	weighted_adj = False
	dimensions = config['dimensions']
	impute_list = config['impute_list']
	res = config['resolution']
	neighbor_num = config['neighbor_num'] + 1
	config_name = config['config_name']
	mode = config["loss_mode"]
	print ("loss mode", mode)
	if mode == 'rank':
		rank_thres = config['rank_thres']
	embedding_name = config['embedding_name']
	
	if "coassay" in config:
		coassay = config['coassay']
	else:
		coassay = False
	
	
	save_path = os.path.join(temp_dir, "model")
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	save_path = os.path.join(save_path, 'model.chkpt')
	
	impute_no_nbr_flag = True
	if "impute_no_nbr" in config:
		impute_no_nbr_flag = config['impute_no_nbr']
		
		
	impute_with_nbr_flag = True
	if "impute_with_nbr" in config:
		impute_with_nbr_flag = config['impute_with_nbr']
	
	
	if "embedding_epoch" in config:
		embedding_epoch = config['embedding_epoch']
	else:
		embedding_epoch = 60
		
	if "no_nbr_epoch" in config:
		no_nbr_epoch = config['no_nbr_epoch']
	else:
		no_nbr_epoch = 45
	if "with_nbr_epoch" in config:
		with_nbr_epoch = config['with_nbr_epoch']
	else:
		with_nbr_epoch = 30
	
	remove_be_flag = False
	if "correct_be_impute" in config:
		if config['correct_be_impute']:
			remove_be_flag = True
	precompute_weighted_nbr = True
	if  "precompute_weighted_nbr" in config:
		if not config["precompute_weighted_nbr"]:
			precompute_weighted_nbr = False
	
	# All above are just loading parameters from the config file
	
	#Training related parameters, but these are hard coded as they usually don't require tuning
	# collect_num=x means, one cpu thread would collect x batches of samples
	collect_num = 1
	update_num_per_training_epoch = 1000
	update_num_per_eval_epoch = 10
	
	# Load everything from the hdf5 file
	with h5py.File(os.path.join(temp_dir, "node_feats.hdf5"), "r") as input_f:
		num = np.array(input_f['num'])
		train_data, train_weight, train_chrom = [], [], []
		test_data, test_weight, test_chrom = [], [], []
		
		for c in range(len(chrom_list)):
			train_data.append(np.array(input_f['train_data_%s' % chrom_list[c]]).astype('int'))
			train_weight.append(np.array(input_f['train_weight_%s' % chrom_list[c]]).astype('float32'))
			train_chrom.append(np.array(input_f['train_chrom_%s' % chrom_list[c]]).astype('int8'))
			
			test_data.append(np.array(input_f['test_data_%s' % chrom_list[c]]).astype('int'))
			test_weight.append(np.array(input_f['test_weight_%s' % chrom_list[c]]).astype('float32'))
			test_chrom.append(np.array(input_f['test_chrom_%s' % chrom_list[c]]).astype('int8'))
			
		distance2weight = np.array(input_f['distance2weight'])
		total_sparsity_cell = np.array(input_f['sparsity'])
		start_end_dict = np.array(input_f['start_end_dict'])
		cell_feats = np.array(input_f['extra_cell_feats'])
		cell_feats1 = np.array(input_f['cell2weight'])
	
	batch_size = int(256 * max((1000000 / res), 1) * max(num[0] / 6000, 1))
	
	num_list = np.cumsum(num)
	max_bin = int(np.max(num[1:]))
	cell_num = num[0]
	cell_ids = (torch.arange(num[0])).long().to(device, non_blocking=True)
	
	mem_efficient_flag = cell_num > 30

	distance2weight = distance2weight.reshape((-1, 1))
	distance2weight = torch.from_numpy(distance2weight).float().to(device, non_blocking=True)
	
	
	# if mode == 'regression':
	# 	# normalize by cell
	# 	print ("normalize by cell")
	# 	for i in trange(cell_num):
	# 		cell = i + 1
	# 		weight[data[:, 0] == cell] /= (np.sum(weight[data[:, 0] == cell]) / 10000)
	#
	
	
	total_possible = 0
	chrom_start_end = np.load(os.path.join(temp_dir, "chrom_start_end.npy"))
	for c in chrom_start_end:
		start, end = c[0], c[1]
		for i in range(start, end):
			for j in range(i, end):
				total_possible += 1
	
	total_possible *= cell_num
	sparsity = np.sum([len(d)+len(d2) for d, d2 in zip(train_data, test_data)]) / total_possible
	
	
	
	# total_sparsity_cell = np.load(os.path.join(temp_dir, "sparsity.npy"))
	median_total_sparsity_cell = np.median(total_sparsity_cell)
	print ("total_sparsity_cell", median_total_sparsity_cell, sparsity)
	# Trust the original cell feature more if there are more non-zero entries for faster convergence.
	# If there are more than 95% zero entries at 1Mb, then rely more on Higashi non-linear transformation
	if np.median(total_sparsity_cell) >= 0.05:
		print ("contractive loss")
		contractive_flag = True
		contractive_loss_weight = 1e-3
	else:
		print("no contractive loss")
		contractive_flag = False
		contractive_loss_weight = 0.0
	
	# Dependes on the sparsity, change the number of negative samples
	if sparsity > 0.3:
		neg_num = 1
	elif sparsity > 0.2:
		neg_num = 2
	elif sparsity > 0.15:
		neg_num = 3
	elif sparsity > 0.1:
		neg_num = 4
	else:
		neg_num = 5
		
	batch_size *= (1 + neg_num)
	
	
	print("Node type num", num, num_list)
	start_end_dict = np.concatenate([np.zeros((1, 2)), start_end_dict], axis=0).astype('int')



	cell_feats = np.sum(cell_feats, axis=-1, keepdims=True)
	cell_feats1 = torch.from_numpy(cell_feats1).float().to(device, non_blocking=True)
	cell_feats = np.log1p(cell_feats)

	if "batch_id" in config or "library_id" in config:
		label_info = pickle.load(open(os.path.join(data_dir, "label_info.pickle"), "rb"))

		if "batch_id" in config:
			label = np.array(label_info[config["batch_id"]])
		else:
			label = np.array(label_info[config["library_id"]])
		uniques = np.unique(label)
		target2int = np.zeros((len(label), len(uniques)), dtype='float32')
		
		for i, t in enumerate(uniques):
			target2int[label == t, i] = 1
		print (target2int.shape)
		cell_feats = np.concatenate([cell_feats, target2int], axis=-1)
	cell_feats = np.concatenate([np.zeros((1, cell_feats.shape[-1])), cell_feats], axis=0).astype('float32')

	embeddings_initial, attribute_dict, targets_initial = generate_attributes()
	
	
	sparse_chrom_list = np.load(os.path.join(temp_dir, "sparse_nondiag_adj_nbr_1.npy"), allow_pickle=True)
	for i in range(len(sparse_chrom_list)):
		for j in range(len(sparse_chrom_list[i])):
			sparse_chrom_list[i][j] = sparse_chrom_list[i][j].astype('float32')
	
	if not mem_efficient_flag:
		import copy
		sparse_chrom_list_GCN = sparse_chrom_list
		sparse_chrom_list_dict = copy.deepcopy(sparse_chrom_list)
		conv_weight = torch.ones((1, 1, 3, 3)).float()
		for chrom in range(len(sparse_chrom_list)):
			for cell in range(len(sparse_chrom_list[0])):
				sparse_chrom_list_dict[chrom][cell] = np.array(sparse_chrom_list_dict[chrom][cell].todense())
				m = sparse_chrom_list_dict[chrom][cell][None, None, :, :]
				m = torch.from_numpy(m).float()
				m = F.conv2d(m, conv_weight, padding=2)
				sparse_chrom_list_dict[chrom][cell] = m.detach().cpu().numpy()[0, 0, 1:-1, 1:-1]
	else:
		sparse_chrom_list_GCN = sparse_chrom_list
		sparse_chrom_list_dict = sparse_chrom_list
	

		
	if mode == 'classification':
		train_weight_mean = np.mean(train_weight)
		
		train_weight, test_weight = transform_weight_class(train_weight, train_weight_mean, neg_num), \
									transform_weight_class(test_weight, train_weight_mean, neg_num)
	elif mode == 'rank':
		train_weight += 1
		test_weight += 1
	
	
	# Constructing the model
	node_embedding_init = MultipleEmbedding(
		embeddings_initial,
		dimensions,
		False,
		num_list, targets_initial).to(device, non_blocking=True)
	if training_stage <= 1:
		node_embedding_init.wstack[0].fit(embeddings_initial[0], 300, sparse=False, targets=torch.from_numpy(targets_initial[0]).float().to(device, non_blocking=True), batch_size=1024)
	
	
	higashi_model = Hyper_SAGNN(
		n_head=8,
		d_model=dimensions,
		d_k=16,
		d_v=16,
		diag_mask=True,
		bottle_neck=dimensions,
		attribute_dict=attribute_dict,
		cell_feats=cell_feats,
		encoder_dynamic_nn=node_embedding_init,
		encoder_static_nn=node_embedding_init,
		chrom_num = len(chrom_list)).to(device, non_blocking=True)

	loss = F.binary_cross_entropy_with_logits
	
	for name, param in higashi_model.named_parameters():
		print(name, param.requires_grad, param.shape)
	
	optimizer = torch.optim.Adam(list(higashi_model.parameters()) + list(node_embedding_init.parameters()),
								  lr=1e-3)
	
	scheduler = ReduceLROnPlateau(
		optimizer,
		patience=3,
		factor=0.8,
		threshold=1e-3,
		min_lr=1e-6,
		threshold_mode="abs",
		verbose=True,
	)
	
	model_parameters = filter(lambda p: p.requires_grad, higashi_model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print("params to be trained", params)
	alpha = 1.0
	beta = 1e-2

	
	
	pair_ratio = 0.0
	dynamic_pair_ratio = False

	cell_neighbor_list = [[i] for i in range(num[0] + 1)]
	cell_neighbor_weight_list = [[1] for i in range(num[0] + 1)]
	

	training_data_generator = DataGenerator(train_data, train_chrom, train_weight,
											int(batch_size / (neg_num + 1) * collect_num),
											True, num_list, k=collect_num)
	validation_data_generator = DataGenerator(test_data, test_chrom, test_weight,
											  int(batch_size / (neg_num + 1)),
											  False, num_list, k=1)
	
	steps = 1
	# First round, no cell dependent GNN
	if training_stage <= 1:
		# print ("Pre-training")
		# use_recon = False
		# higashi_model.only_distance=True
		# train(higashi_model,
		#       loss=loss,
		#       training_data_generator=training_data_generator,
		#       validation_data_generator=validation_data_generator,
		#       optimizer=[optimizer], epochs=5,
		#       load_first=False, save_embed=False)
		
		
		pair_ratio = 0.0
		if mem_efficient_flag:
			dynamic_pair_ratio = True
		else:
			dynamic_pair_ratio = False

		# Training Stage 1
		scheduler = ReduceLROnPlateau(
			optimizer,
			patience=6,
			factor=0.9,
			threshold=1e-3,
			min_lr=1e-6,
			threshold_mode="abs",
			verbose=True,
		)

		print ("First stage training")
		higashi_model.only_distance = False
		use_recon = True
		train(higashi_model,
			  loss=loss,
			  training_data_generator=training_data_generator,
			  validation_data_generator=validation_data_generator,
			  optimizer=[optimizer], epochs=embedding_epoch,
			  load_first=False, save_embed=True, save_name="_stage1")


		checkpoint = {
			'model_link': higashi_model.state_dict()}

		torch.save(checkpoint, save_path+"_stage1")


	# Loading Stage 1
	checkpoint = torch.load(save_path+"_stage1", map_location=current_device)
	higashi_model.load_state_dict(checkpoint['model_link'])
	save_embeddings(higashi_model)
	node_embedding_init.off_hook([0])

	max_distance = config['maximum_distance']
	if max_distance < 0:
		max_bin = int(1e5)
	else:
		max_bin = int(max_distance / res)

	min_distance = config['minimum_distance']
	if min_distance < 0:
		min_bin = 0
	else:
		min_bin = int(min_distance / res)

	if training_stage <= 3:
		training_data_generator.filter_edges(min_bin, max_bin)
		validation_data_generator.filter_edges(min_bin, max_bin)

	alpha = 1.0
	beta = 1e-3
	dynamic_pair_ratio = False
	use_recon = False
	contractive_flag = False
	contractive_loss_weight = 0.0
	
	if mem_efficient_flag:
		pair_ratio = 0.5
	else:
		pair_ratio = 0.0

	remove_flag = True
	node_embedding2 = GraphSageEncoder_with_weights(features=node_embedding_init, linear_features=node_embedding_init,
					 								feature_dim=dimensions,
													embed_dim=dimensions,
													num_sample=8, gcn=False, num_list=num_list,
													transfer_range=0, start_end_dict=start_end_dict,
													pass_pseudo_id=False, remove=remove_flag,
													pass_remove=False).to(device, non_blocking=True)


	higashi_model.encode1.dynamic_nn = node_embedding2

	optimizer = torch.optim.Adam(list(higashi_model.parameters()) + list(node_embedding_init.parameters()),
	                              lr=1e-3)
	
	scheduler = ReduceLROnPlateau(
		optimizer,
		patience=3,
		factor=0.8,
		threshold=1e-3,
		min_lr=1e-6,
		threshold_mode="abs",
		verbose=True,
	)

	if impute_no_nbr_flag or impute_with_nbr_flag:

		# Second round, with cell dependent GNN, but no neighbors
		if training_stage <= 2:
			steps = 2
			# Training Stage 2
			print("Second stage training")
			train(higashi_model,
				  loss=loss,
				  training_data_generator=training_data_generator,
				  validation_data_generator=validation_data_generator,
				  optimizer=[optimizer], epochs=no_nbr_epoch,
				  load_first=False, save_embed=False,
			      save_name="_stage2")

			checkpoint = {
					'model_link': higashi_model.state_dict()}

			torch.save(checkpoint, save_path + "_stage2")

	# 	# Loading Stage 2
		checkpoint = torch.load(save_path + "_stage2", map_location=current_device)
		higashi_model.load_state_dict(checkpoint['model_link'])




	if training_stage <= 2:
		# Impute Stage 2

		if impute_no_nbr_flag:
			if non_para_impute:
				cell_id_all = [np.arange(num[0])]
				impute_process(args.config, higashi_model, "%s_nbr_%d_impute"  % (embedding_name, 0), mode, 0, num[0], os.path.join(temp_dir, "sparse_nondiag_adj_nbr_1.npy"))
			else:
				torch.save(higashi_model, save_path + "_stage2_model")
				cell_id_all = np.arange(num[0])
				cell_id_all = np.array_split(cell_id_all, gpu_num-1)
				select_gpus = get_free_gpu(gpu_num - 1)
				for i in range(gpu_num-1):
					impute_pool.submit(mp_impute, args.config,
					                   save_path + "_stage2_model",
					                   "%s_nbr_%d_impute_part_%d" %(embedding_name, 0, i),
					                   mode, np.min(cell_id_all[i]),
					                   np.max(cell_id_all[i]) + 1,
					                   os.path.join(temp_dir, "sparse_nondiag_adj_nbr_1.npy"),
					                   None,
					                   select_gpus[i])
					# time.sleep(30)
				impute_pool.shutdown(wait=True)

				impute_pool = ProcessPoolExecutor(max_workers=gpu_num)
				linkhdf5("%s_nbr_%d_impute" % (embedding_name, 0), cell_id_all, temp_dir, impute_list, None)

	if impute_with_nbr_flag:
		nbr_mode = 0
		if neighbor_num == 1:
			print ("cannot train step 3 with neighbor_num = 0")
			raise EOFError
		if not impute_no_nbr_flag:
			nbr_mode = 0
		if remove_be_flag:
			nbr_mode = 0

		# Training Stage 3
		print ("getting cell nbr's nbr list")

		if remove_be_flag and ("batch_id" in config):
			cell_neighbor_list, cell_neighbor_weight_list = get_cell_neighbor_be(nbr_mode)
		else:
			cell_neighbor_list, cell_neighbor_weight_list = get_cell_neighbor(nbr_mode)

		weight_dict = {}
		print (cell_neighbor_list[:10], cell_neighbor_weight_list[:10])

		for i in trange(len(cell_neighbor_list)):
			for c, w in zip(cell_neighbor_list[i], cell_neighbor_weight_list[i]):
				weight_dict[(c, i)] = w
		weighted_adj = True

		if precompute_weighted_nbr:
			new_sparse_chrom_list = [[] for i in range(len(sparse_chrom_list))]
			for c, chrom in enumerate(chrom_list):
				new_cell_chrom_list = []
				for cell in np.arange(num_list[0]) + 1:
					mtx = 0
					for nbr_cell in cell_neighbor_list[cell]:
						balance_weight = weight_dict[(nbr_cell, cell)]
						mtx = mtx + balance_weight * sparse_chrom_list[c][nbr_cell - 1]
					mtx = csr_matrix(mtx)
					new_cell_chrom_list.append(mtx)
				new_cell_chrom_list = np.array(new_cell_chrom_list)
				new_sparse_chrom_list[c] = new_cell_chrom_list
			new_sparse_chrom_list = np.array(new_sparse_chrom_list)
			sparse_chrom_list_GCN = new_sparse_chrom_list
			
		np.save(os.path.join(temp_dir, "weighted_info.npy"), np.array([cell_neighbor_list, weight_dict]), allow_pickle=True)

		# node_embedding1 = GraphSageEncoder_with_weights(features=node_embedding2, linear_features=node_embedding2,
		#                                                 feature_dim=dimensions,
		#                                                 embed_dim=dimensions, node2nbr=neighbor_list,
		#                                                 num_sample=16, gcn=False, num_list=num_list,
		#                                                 transfer_range=0, start_end_dict=start_end_dict,
		#                                                 pass_pseudo_id=True, remove=remove_flag,
		#                                                 pass_remove=False).to(device, non_blocking=True)
		#
		# higashi_model.encode1.dynamic_nn = node_embedding1
		
		# higashi_model = torch.load(save_path + "_stage2_model", map_location=current_device)
		# embeddings_initial = higashi_model.encode1.static_nn
		print (node_embedding_init.on_hook_set)
		print (higashi_model.encode1.dynamic_nn.fix)
		
		optimizer = torch.optim.Adam(higashi_model.parameters(), lr=1e-3)
		
		scheduler = ReduceLROnPlateau(
			optimizer,
			patience=3,
			factor=0.8,
			threshold=1e-3,
			min_lr=1e-6,
			threshold_mode="abs",
			verbose=True,
		)


		if training_stage <= 3:
			steps = 3
			print("Final stage training")
			train(higashi_model,
				  loss=loss,
				  training_data_generator=training_data_generator,
				  validation_data_generator=validation_data_generator,
				  optimizer=[optimizer], epochs=with_nbr_epoch,
				  load_first=False, save_name="_stage3", save_embed=False)

			checkpoint = {
				'model_link': higashi_model.state_dict()}

			torch.save(checkpoint, save_path+"_stage3")

			del training_data_generator, validation_data_generator
		# Loading Stage 3
		checkpoint = torch.load(save_path + "_stage3", map_location=current_device)
		higashi_model.load_state_dict(checkpoint['model_link'])



		del sparse_chrom_list, weight_dict



		# Impute Stage 3
		if impute_with_nbr_flag:
			if non_para_impute:
				cell_id_all = [np.arange(num[0])]
				impute_process(args.config, higashi_model, "%s_nbr_%d_impute"  % (embedding_name, neighbor_num-1), mode, 0, num[0], os.path.join(temp_dir, "sparse_nondiag_adj_nbr_1.npy" ), os.path.join(temp_dir, "weighted_info.npy"))
			else:
				select_gpus = get_free_gpu(gpu_num - 1)
				print ("select gpus", select_gpus)
				torch.save(higashi_model, save_path + "_stage3_model")
				cell_id_all = np.arange(num[0])
				cell_id_all = np.array_split(cell_id_all, gpu_num-1)
				for i in range(gpu_num-2, -1, -1):
					impute_pool.submit(mp_impute, args.config,
					                   save_path + "_stage3_model",
					                   "%s_nbr_%d_impute_part_%d" %(embedding_name, neighbor_num-1, i),
					                   mode, np.min(cell_id_all[i]),
					                   np.max(cell_id_all[i]) + 1,
					                   os.path.join(temp_dir, "sparse_nondiag_adj_nbr_1.npy"),
					                   os.path.join(temp_dir, "weighted_info.npy"),
					                   select_gpus[i])
					# time.sleep(30)

				impute_pool.shutdown(wait=True)

				extra_str = "%s_nbr_%d_impute" % (embedding_name, 0) if (impute_no_nbr_flag and nbr_mode == 1 and not remove_be_flag) else None

				# When the 1nb imputation is there and nbr_mode=1 (itself is not included during learning), add the predicted values with only 1nb to the neighbor version.
				linkhdf5("%s_nbr_%d_impute" % (embedding_name, neighbor_num-1), cell_id_all, temp_dir, impute_list, extra_str)

