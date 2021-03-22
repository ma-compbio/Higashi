import multiprocessing as mp
import warnings
from Higashi_backend.Modules import *
from Higashi_backend.Functions import *
from Higashi_backend.utils import *
from Impute import impute_process
import argparse
import resource

from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, MinMaxScaler
import pickle
import subprocess

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
		ids = np.where(memory_available == max_mem)[0]
		if num == 1:
			chosen_id = int(np.random.choice(ids, 1)[0])
			print("setting to gpu:%d" % chosen_id)
			torch.cuda.set_device(chosen_id)
			return "cuda:%d" % chosen_id
		else:
			return np.random.choice(ids, num)
		
	else:
		return


def forward_batch_hyperedge(model, loss_func, batch_data, batch_weight, batch_chrom, y):
	x = batch_data
	w = batch_weight
	pred = model(x, batch_chrom)
	
	if use_recon:
		adj = node_embedding_init.embeddings[0](cell_ids)
		targets = node_embedding_init.targets[0](cell_ids)
		embed, recon = node_embedding_init.wstack[0](adj, return_recon=True)
		mse_loss = F.mse_loss(recon, targets)
		# mse_loss = XSigmoidLoss(recon, targets)
	else:
		mse_loss = torch.as_tensor([0], dtype=torch.float).to(device)
		
		
	if mode == 'classification':
		label = y
		main_loss = loss_func(pred, y, weight=w)
		
	elif mode == 'rank':
		pred = F.softplus(pred).float()
		diff = (pred.view(-1, 1) - pred.view(1, -1)).view(-1)
		diff_w = (w.view(-1, 1) - w.view(1, -1)).view(-1)
		mask_rank = torch.abs(diff_w) > rank_thres
		diff = diff[mask_rank].float()
		diff_w = diff_w[mask_rank]
		label = (diff_w > 0).float()
		# main_loss = torch.mean(torch.clamp(- diff * label + margin, min=0.0))
		main_loss = loss_func(diff, label)
		
		if not use_recon:
			if neg_num > 0:
				mask_w_eq_zero = w == 0
				makeitzero = F.mse_loss(w[mask_w_eq_zero], pred[mask_w_eq_zero])
				mse_loss += makeitzero
				
	elif mode == 'regression':
		label = w
		main_loss =  F.mse_loss(pred, w)
	else:
		print ("wrong mode")
		raise EOFError
	
	return pred, main_loss, mse_loss


def train_epoch(model, loss_func, training_data_generator, optimizer_list):
	# Epoch operation in training phase
	# Simultaneously train on : hyperedge-prediction (1)
	loss_func = loss_func
	
	model.train()
	
	bce_total_loss = 0
	mse_total_loss = 0
	final_batch_num = 0
	
	batch_num = int(update_num_per_training_epoch / collect_num)
	
	pool = ProcessPoolExecutor(max_workers=cpu_num)
	p_list = []
	y_list, pred_list = [], []
	
	bar = trange(batch_num * collect_num, desc=' - (Training) ', leave=False, )
	for i in range(batch_num):
		edges_part, edges_chrom, edge_weight_part = training_data_generator.next_iter()
		p_list.append(pool.submit(one_thread_generate_neg, edges_part, edges_chrom, edge_weight_part))
	
	for p in as_completed(p_list):
		batch_edge_big, batch_y_big, batch_edge_weight_big, batch_chrom_big = p.result()
		batch_edge_big = np2tensor_hyper(batch_edge_big, dtype=torch.long)
		batch_y_big, batch_edge_weight_big = torch.from_numpy(batch_y_big), torch.from_numpy(batch_edge_weight_big)
		
		batch_edge_big, batch_y_big, batch_edge_weight_big = batch_edge_big.to(device), batch_y_big.to(
			device), batch_edge_weight_big.to(device)
		size = int(len(batch_edge_big) / collect_num)
		for j in range(collect_num):
			batch_edge, batch_edge_weight, batch_y, batch_chrom = batch_edge_big[j * size: min((j + 1) * size, len(batch_edge_big))], \
													 batch_edge_weight_big[
													 j * size: min((j + 1) * size, len(batch_edge_big))], \
													 batch_y_big[j * size: min((j + 1) * size, len(batch_edge_big))], \
													 batch_chrom_big[j * size: min((j + 1) * size, len(batch_edge_big))]
			
			pred, loss_bce, loss_mse = forward_batch_hyperedge(model, loss_func, batch_edge,
																			batch_edge_weight, batch_chrom, y=batch_y)
			
			
			y_list.append(batch_y)
			pred_list.append(pred)
			
			final_batch_num += 1
			
			for opt in optimizer_list:
				opt.zero_grad()
			loss_bce.backward(retain_graph=True)
			main_norm = node_embedding_init.wstack[0].weight_list[0].grad.data.norm(2)
			
			if use_recon:
				for opt in optimizer_list:
					opt.zero_grad()
				loss_mse.backward(retain_graph=True)

				recon_norm = node_embedding_init.wstack[0].weight_list[0].grad.data.norm(2)
				ratio = beta * main_norm / recon_norm
				ratio1 = max(ratio, 100 * np.median(total_sparsity_cell) - 8)
				
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
				opt.zero_grad()
			
			# backward
			train_loss.backward()
			
			# update parameters
			for opt in optimizer_list:
				opt.step()
			bar.update(n=1)
			bar.set_description(" - (Training) BCE:  %.3f MSE: %.3f Loss: %.3f norm_ratio: %.2f"  %
								(loss_bce.item(), loss_mse.item(),  train_loss.item(), ratio),
								refresh=False)
			
			bce_total_loss += loss_bce.item()
			mse_total_loss += loss_mse.item()
		p_list.remove(p)
		del p
	
	y = torch.cat(y_list)
	pred = torch.cat(pred_list)
	auc1, auc2 = roc_auc_cuda(y, pred)
	pool.shutdown(wait=True)
	bar.close()
	return bce_total_loss / final_batch_num, mse_total_loss / final_batch_num, accuracy(
		y, pred), auc1, auc2


def eval_epoch(model, loss_func, validation_data_generator):
	"""Epoch operation in evaluation phase"""
	bce_total_loss = 0
	
	model.eval()
	with torch.no_grad():
		pred, label = [], []
		auc1_list, auc2_list = [], []
		for i in tqdm(range(update_num_per_eval_epoch), desc='  - (Validation)   ', leave=False):
			edges_part, edges_chrom, edge_weight_part = validation_data_generator.next_iter()
			
			batch_x, batch_y, batch_w, batch_chrom = one_thread_generate_neg(edges_part, edges_chrom, edge_weight_part)
			batch_x = np2tensor_hyper(batch_x, dtype=torch.long)
			batch_y, batch_w = torch.from_numpy(batch_y), torch.from_numpy(batch_w)
			batch_x, batch_y, batch_w = batch_x.to(device), batch_y.to(device), batch_w.to(device)
			
			pred_batch, eval_loss, _ = forward_batch_hyperedge(model, loss_func, batch_x, batch_w, batch_chrom, y=batch_y)
			
			pred.append(pred_batch)
			label.append(batch_y)
			bce_total_loss += eval_loss.item()
			
			auc1, auc2 = roc_auc_cuda(batch_y, pred_batch)
			auc1_list.append(auc1)
			auc2_list.append(auc2)
		
		pred = torch.cat(pred, dim=0)
		label = torch.cat(label, dim=0)
		
		acc = accuracy(pred, label)
	
	return bce_total_loss / (i + 1), acc, np.mean(auc1_list), np.mean(auc2_list)


def check_nonzero(x, c):
	# minus 1 because add padding indexå
	dim = sparse_chrom_list[c][x[0]-1].shape[-1]
	return sparse_chrom_list[c][x[0]-1][max(0, x[1]-1-num_list[c]-1):min(x[1]-1-num_list[c]+2, dim-1), max(0, x[2]-1-num_list[c]-1):min(x[2]-1-num_list[c]+2, dim-1)].sum() > 0

	
def generate_negative_cpu(x, x_chrom, forward=True):
	global pair_ratio
	rg = np.random.default_rng()
	
	neg_list, neg_chrom = [], []
	
	if forward:
		func1 = pass_
	else:
		func1 = tqdm
	
	# Why neg_num + 1? Because sometimes it may fails to find samples after certain trials.
	# So we just add 1 fore more trials in the first place
	
	change_list_all = rg.integers(0, x.shape[-1], (len(x), neg_num))
	simple_or_hard_all = rg.random((len(x), neg_num + 1))
	for j, sample in enumerate(func1(x)):
		
		for i in range(neg_num):
			temp = np.copy(sample)
			change = change_list_all[j, i]
			trial = 0
			while check_nonzero(temp, x_chrom[j]):
				temp = np.copy(sample)
				
				# Try too many times on one samples, move on
				trial += 1
				if trial >= 500:
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
				if ((temp[2] - temp[1]) >= max_bin) or ((temp[2] - temp[1]) < min_bin):
					temp = np.copy(sample)
				
				
			
			if len(temp) > 0:
				neg_list.append(temp)
				neg_chrom.append(x_chrom[j])
	
	return neg_list, neg_chrom


def one_thread_generate_neg(edges_part, edges_chrom, edge_weight):
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
	
	index = np.random.permutation(len(x))
	x, y, w, x_chrom = x[index], y[index], w[index], x_chrom[index]
		
	
	if isinstance(higashi_model.encode1.dynamic_nn, GraphSageEncoder_with_weights):
		cell_ids = np.stack([x[:, 0], x[:, 0]], axis=-1).reshape((-1))
		bin_ids = x[:, 1:].reshape((-1))
		nodes_chrom = np.stack([x_chrom, x_chrom], axis=-1).reshape((-1))
		to_neighs = []
		
		for c, cell_id, bin_id in zip(nodes_chrom, cell_ids, bin_ids):
			row = new_sparse_chrom_list[c][cell_id - 1][bin_id - 1 - num_list[c]]
			
			
			nbrs = row.nonzero()
			
			nbr_value = np.array(
				row.data).reshape((-1))
			nbrs = np.array(nbrs[1]).reshape((-1)) + 1 + num_list[c]
			

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
		
	else:
		to_neighs = x_chrom
	
	return x, y, w, to_neighs


def train(model, loss, training_data_generator, validation_data_generator, optimizer, epochs, load_first, save_embed=False, save_name=""):
	global pair_ratio
	no_improve = 0
	if load_first:
		checkpoint = torch.load(save_path+save_name)
		model.load_state_dict(checkpoint['model_link'])
	
	valid_accus = [0]
	train_accus = []

	
	
	for epoch_i in range(epochs):
		if save_embed:
			save_embeddings(model)
		
		print('[ Epoch', epoch_i, 'of', epochs, ']')
		
		start = time.time()
		
		bce_loss, mse_loss, train_accu, auc1, auc2 = train_epoch(
			model, loss, training_data_generator, optimizer)
		print('  - (Training)   bce: {bce_loss: 7.4f}, mse: {mse_loss: 7.4f}, '
			  ' acc: {accu:3.3f} %, auc: {auc1:3.3f}, aupr: {auc2:3.3f}, '
			  'elapse: {elapse:3.3f} s'.format(
			bce_loss=bce_loss,
			mse_loss=mse_loss,
			accu=100 *
				 train_accu,
			auc1=auc1,
			auc2=auc2,
			elapse=(time.time() - start)))
		
		start = time.time()
		valid_bce_loss, valid_accu, valid_auc1, valid_auc2 = eval_epoch(model, loss, validation_data_generator)
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
		
		
		
		# Dynamic pair ratio for stage one
		if dynamic_pair_ratio:
			if pair_ratio < 0.5:
				pair_ratio += 0.1
			elif pair_ratio > 0.5:
				pair_ratio = 0.5
			print("pair_ratio", pair_ratio)
		
		# if (not dynamic_pair_ratio) or pair_ratio == 0.5:
		# 	valid_accus += [valid_auc2]
		#
		# 	# if valid_auc2 >= max(valid_accus):
		# 	# 	print("%.2f to %.2f saving" % (valid_auc2, float(max(valid_accus))))
		# 	# 	torch.save(checkpoint, save_path+save_name)
		# 	if len(train_accus) > 0:
		# 		if bce_loss <= (min(train_accus) - 1e-3):
		# 			no_improve = 0
		# 			train_accus += [bce_loss]
		# 		else:
		# 			print(bce_loss, min(train_accus) - 1e-3)
		# 			no_improve += 1
		if epoch_i % 5 == 0:
			checkpoint = {
				'model_link': model.state_dict(),
				'epoch': epoch_i}
			torch.save(checkpoint, save_path + save_name)
	start = time.time()
	valid_bce_loss, valid_accu, valid_auc1, valid_auc2 = eval_epoch(model, loss, validation_data_generator)
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
		ids = torch.arange(1, num_list[-1] + 1).long().to(device).view(-1)
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
					old_static = np.load(os.path.join(temp_dir, "%s_%d_origin.npy" % (embedding_name, i)))
					update_rate = np.sum((old_static - static) ** 2, axis=-1) / np.sum(old_static ** 2, axis=-1)
					print("update_rate: %f\t%f" % (np.min(update_rate), np.max(update_rate)))
				except Exception as e:
					pass
				
			np.save(os.path.join(temp_dir, "%s_%d_origin.npy" % (embedding_name, i)), static)
	
	torch.cuda.empty_cache()
	return embeddings

		
def generate_attributes():
	embeddings = []
	targets = []
	pca_after = []
	
	for c in chrom_list:
		a = np.load(os.path.join(temp_dir, "%s_cell_PCA.npy" % c))
		pca_after.append(a)
	
	if coassay:
		print ("coassay")
		cell_attributes = np.load(os.path.join(temp_dir, "pretrain_coassay.npy")).astype('float32')
		# cell_targets = np.load(os.path.join(temp_dir, "coassay_all.npy")).astype('float32')
		# cell_targets = StandardScaler().fit_transform(cell_targets)
		# cell_attributes = StandardScaler().fit_transform(cell_attributes)
		targets.append(cell_attributes)
		pca_after = StandardScaler().fit_transform(pca_after)
		embeddings.append(pca_after)
	else:
		# print (pca_after)
		pca_after1 = remove_BE_linear(pca_after, config, data_dir)
		pca_after1 = StandardScaler().fit_transform(pca_after1.reshape((-1, 1))).reshape((len(pca_after1), -1))
		pca_after2 = [StandardScaler().fit_transform(x) for x in pca_after]
		pca_after2 = remove_BE_linear(pca_after2, config, data_dir)
		pca_after2 = StandardScaler().fit_transform(pca_after2.reshape((-1, 1))).reshape((len(pca_after2), -1))

		targets.append(pca_after2.astype('float32'))
		embeddings.append(pca_after1.astype('float32'))
		


	
	
	for i, c in enumerate(chrom_list):
		temp = np.load(os.path.join(temp_dir, "%s_bin_adj.npy" % c)).astype('float32')
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
		attribute_all.append(attribute)
	
	attribute_all = np.concatenate(attribute_all, axis=0)
	attribute_dict = np.concatenate([np.zeros((num[0] + 1, attribute_all.shape[-1])), attribute_all], axis=0).astype(
		'float32')
	# attribute_dict = StandardScaler().fit_transform(attribute_dict)
	return embeddings, attribute_dict, targets


def get_cell_neighbor_be(start=1):
	# save_embeddings(higashi_model, True)
	v = np.load(os.path.join(temp_dir, "%s_0_origin.npy" % embedding_name))
	
	
	distance = pairwise_distances(v, metric='euclidean')
	distance_sorted = np.sort(distance, axis=-1)
	distance /= np.quantile(distance_sorted[:, 1:neighbor_num].reshape((-1)), q=0.25)
	# distance /= np.mean(distance)
	cell_neighbor_list = [[] for i in range(num[0] + 1)]
	cell_neighbor_weight_list = [[] for i in range(num[0] + 1)]
	if "batch_id" in config:
		label_info = pickle.load(open(os.path.join(data_dir, "label_info.pickle"), "rb"))
		label = np.array(label_info[config["batch_id"]])
		batches = np.unique(label)
		equal_num = int(math.ceil(neighbor_num / (len(batches))))
		
		indexs = [np.where(label == b)[0] for b in batches]
		
		for i, d in enumerate(tqdm(distance)):
			neighbor  = []
			weight = []
			b_this = label[i]
			for j in range(len(batches)):
				neighbor.append(indexs[j][np.argsort(d[indexs[j]])][:equal_num])
				weight.append(d[neighbor[-1]])
				
			weight = [w/np.mean(w) for w in weight]
			neighbor = np.concatenate(neighbor)
			weight = np.concatenate(weight)
			
			neighbor = neighbor[np.argsort(weight)]
			weight = np.sort(weight)
			
			index = np.random.permutation(len(neighbor)-1)
			neighbor[1:] = neighbor[1:][index]
			weight[1:] = weight[1:][index]
			neighbor = neighbor[:neighbor_num]
			weight = weight[:neighbor_num]
			# neighbor = neighbor[index][:neighbor_num]
			# weight = weight[index][:neighbor_num]
			neighbor = neighbor[np.argsort(weight)]
			weight = np.sort(weight)

		
			new_w = np.exp(-weight[start:])
			new_w /= np.sum(new_w)
			# new_w[0] = 0.0
			cell_neighbor_list[i + 1] = (neighbor + 1)[start:]
			cell_neighbor_weight_list[i + 1] = (new_w)
	# cell_neighbor_weight_list[i + 1] /= np.sum(cell_neighbor_weight_list[i + 1])
	
	return np.array(cell_neighbor_list), np.array(cell_neighbor_weight_list)


def get_cell_neighbor(start=1):
	# save_embeddings(higashi_model, True)
	v = np.load(os.path.join(temp_dir, "%s_0_origin.npy" % embedding_name))
	distance = pairwise_distances(v, metric='euclidean')
	distance_sorted = np.sort(distance, axis=-1)
	distance /= np.quantile(distance_sorted[:, 1:neighbor_num].reshape((-1)), q=0.25)
	# distance /= np.mean(distance)
	cell_neighbor_list = [[] for i in range(num[0] + 1)]
	cell_neighbor_weight_list = [[] for i in range(num[0] + 1)]
	for i, d in enumerate(tqdm(distance)):
		neighbor = np.argsort(d)[:neighbor_num]
		weight = np.sort(d)[:neighbor_num]
		
		neighbor_new = neighbor
		new_w = weight
		
		new_w = np.exp(-new_w[start:])
		new_w /= np.sum(new_w)
		# new_w[0] = 0.0
		cell_neighbor_list[i + 1] = (neighbor_new + 1)[start:]
		cell_neighbor_weight_list[i + 1] = (new_w)
		# cell_neighbor_weight_list[i + 1] /= np.sum(cell_neighbor_weight_list[i + 1])
	
	
	return np.array(cell_neighbor_list), np.array(cell_neighbor_weight_list)


def mp_impute(config_path, path, name, mode, cell_start, cell_end, sparse_path, weighted_info=None):
	cmd = ["python", "Impute.py", config_path, path, name, mode, str(int(cell_start)), str(int(cell_end)), sparse_path]
	if weighted_info is not None:
		cmd += [weighted_info]
	print (cmd)
	subprocess.call(cmd)

	
if __name__ == '__main__':
	
	# Get parameters from config file
	args = parse_args()
	config = get_config(args.config)
	cpu_num = config['cpu_num']
	if cpu_num < 0:
		cpu_num = int(mp.cpu_count())
	print("cpu_num", cpu_num)
	gpu_num = config['gpu_num']
	print("gpu_num", gpu_num)
	
	if torch.cuda.is_available():
		current_device = get_free_gpu()
	else:
		current_device = 'cpu'
		torch.set_num_threads(cpu_num)
		
	global pair_ratio
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print ("device", device)
	warnings.filterwarnings("ignore")
	rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
	resource.setrlimit(resource.RLIMIT_NOFILE, (3600, rlimit[1]))
	
	
	
	
	training_stage = args.start
	data_dir = config['data_dir']
	temp_dir = config['temp_dir']
	chrom_list = config['chrom_list']
	print (chrom_list)
	
	
	
	if gpu_num < 2:
		non_para_impute = True
		impute_pool = None
	else:
		non_para_impute = False
		impute_pool = ProcessPoolExecutor(max_workers=gpu_num - 1)
		
	weighted_adj = False
	dimensions = config['dimensions']
	impute_list = config['impute_list']
	res = config['resolution']
	neighbor_num = config['neighbor_num']
	local_transfer_range = config['local_transfer_range']
	config_name = config['config_name']
	mode = config["loss_mode"]
	if mode == 'rank':
		rank_thres = config['rank_thres']
	embedding_name = config['embedding_name']
	
	if "coassay" in config:
		coassay = config['coassay']
	else:
		coassay = False
	
	max_distance = config['maximum_distance']
	if max_distance < 0:
		max_bin = 1e5
	else:
		max_bin = int(max_distance / res)
	
	min_distance = config['minimum_distance']
	if min_distance < 0:
		min_bin = 0
	else:
		min_bin = int(min_distance / res)
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
	
	# All above are just loading parameters from the config file
	
	#Training related parameters, but these are hard coded as they usually don't require tuning
	# collect_num=x means, one cpu thread would collect x batches of samples
	collect_num = 10
	update_num_per_training_epoch = 1000
	update_num_per_eval_epoch = 10
	
	batch_size = 96
	
	# Getting how many nodes are there for each node type
	num = np.load(os.path.join(temp_dir, "num.npy"))
	num_list = np.cumsum(num)
	cell_num = num[0]
	cell_ids = (torch.arange(num[0])).long().to(device)
	
	
	# Now start loading data
	data = np.load(os.path.join(temp_dir, "filter_data.npy")).astype('int')
	weight = np.load(os.path.join(temp_dir, "filter_weight.npy")).astype('float32')
	chrom_info = np.load(os.path.join(temp_dir, "filter_chrom.npy")).astype('int')
	
	index = np.arange(len(data))
	np.random.shuffle(index)
	train_index = index[:int(0.85 * len(index))]
	test_index = index[int(0.85 * len(index)):]
	
	if mode == 'regression':
		# normalize by cell
		print ("normalize by cell")
		for cell in trange(cell_num):
			weight[data[:, 0] == cell] /= (np.sum(weight[data[:, 0] == cell]) / 10000)
		weight = StandardScaler().fit_transform(weight.reshape((-1,1))).reshape((-1))
	
	
	total_possible = 0
	chrom_start_end = np.load(os.path.join(temp_dir, "chrom_start_end.npy"))
	for c in chrom_start_end:
		start, end = c[0], c[1]
		for i in range(start, end):
			for j in range(min(i + min_bin, end), min(end, i + max_bin)):
				total_possible += 1
	
	total_possible *= cell_num
	sparsity = len(data) / total_possible
	
	
	
	total_sparsity_cell = np.load(os.path.join(temp_dir, "sparsity.npy"))
	print ("total_sparsity_cell", np.median(total_sparsity_cell), sparsity)
	# Trust the original cell feature more if there are more non-zero entries for faster convergence.
	# If there are more than 95% zero entries at 1Mb, then rely more on Higashi non-linear transformation
	if np.median(total_sparsity_cell) >= 0.05:
		contractive_flag = True
		contractive_loss_weight = 1e-3
	else:
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
	
	
	
	weight += 1
	# if mode == 'rank':
	# 	weight = KBinsDiscretizer(n_bins=50, encode='ordinal', strategy='quantile').fit_transform(
	# 		weight.reshape((-1, 1))).reshape((-1)) + 2
	
	print ("partition into training/test set")
	train_data = data[train_index]
	train_weight = weight[train_index]
	train_chrom = chrom_info[train_index]
	test_data = data[test_index]
	test_weight = weight[test_index]
	test_chrom = chrom_info[test_index]
	print("data", data, np.max(data), data.shape)
	del data, weight, chrom_info
	
	train_mask = ((train_data[:, 2] - train_data[:, 1]) >= min_bin) & ((train_data[:, 2] - train_data[:, 1]) < max_bin)
	train_data = train_data[train_mask]
	train_weight = train_weight[train_mask]
	train_chrom = train_chrom[train_mask]
	
	test_mask = ((test_data[:, 2] - test_data[:, 1]) >= min_bin) & ((test_data[:, 2] - test_data[:, 1]) < max_bin)
	test_data = test_data[test_mask]
	test_weight = test_weight[test_mask]
	test_chrom = test_chrom[test_mask]
	
	
	print("Node type num", num, num_list)
	start_end_dict = np.load(os.path.join(temp_dir, "start_end_dict.npy"))
	start_end_dict = np.concatenate([np.zeros((1, 2)), start_end_dict], axis=0).astype('int')
	
	# Generate features, batch effects related features, etc.
	cell_feats = np.load(os.path.join(temp_dir, "cell_feats.npy")).astype('float32')
	if "batch_id" in config or "library_id" in config:
		label_info = pickle.load(open(os.path.join(data_dir, "label_info.pickle"), "rb"))
		# print (label_info)
		if "batch_id" in config:
			label = np.array(label_info[config["batch_id"]])
		else:
			label = np.array(label_info[config["library_id"]])
		uniques = np.unique(label)
		target2int = np.zeros((len(label), len(uniques)), dtype='float32')
		
		for i, t in enumerate(uniques):
			target2int[label == t, i] = 1
		cell_feats = np.concatenate([cell_feats, target2int], axis=-1)
	cell_feats = np.concatenate([np.zeros((1, cell_feats.shape[-1])), cell_feats], axis=0).astype('float32')
	# print ("cell_feats", cell_feats)
	embeddings_initial, attribute_dict, targets_initial = generate_attributes()
	# print ("attribute_dict.shape", attribute_dict.shape, cell_feats.shape)
	
	
	sparse_chrom_list = np.load(os.path.join(temp_dir, "sparse_nondiag_adj_nbr_1.npy"), allow_pickle=True)
	new_sparse_chrom_list = sparse_chrom_list
	
	if mode == 'classification':
		train_weight_mean = np.mean(train_weight)
		
		train_weight, test_weight = transform_weight_class(train_weight, train_weight_mean, neg_num), \
									transform_weight_class(test_weight, train_weight_mean, neg_num)
	
	# Constructing the model
	node_embedding_init = MultipleEmbedding(
		embeddings_initial,
		dimensions,
		False,
		num_list, targets_initial).to(device)
	node_embedding_init.wstack[0].fit(embeddings_initial[0], 300, sparse=False, targets=torch.from_numpy(targets_initial[0]).float().to(device), batch_size=1024)
	
	
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
		chrom_num = len(chrom_list)).to(device)
	
	loss = F.binary_cross_entropy_with_logits
	
	for name, param in higashi_model.named_parameters():
		print(name, param.requires_grad, param.shape)
	
	optimizer = torch.optim.Adam(list(higashi_model.parameters()) + list(node_embedding_init.parameters()),
								  lr=1e-3)
	
	
	model_parameters = filter(lambda p: p.requires_grad, higashi_model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print("params to be trained", params)
	alpha = 1.0
	beta = 1e-2
	dynamic_pair_ratio = True
	
	
	pair_ratio = 0.0
	cell_neighbor_list = [[i] for i in range(num[0] + 1)]
	cell_neighbor_weight_list = [[1] for i in range(num[0] + 1)]
	

	training_data_generator = DataGenerator(train_data, train_chrom, train_weight,
											int(batch_size / (neg_num + 1) * collect_num),
											True, num_list)
	validation_data_generator = DataGenerator(test_data, test_chrom, test_weight,
											  int(batch_size / (neg_num + 1)),
											  False, num_list)
	
	# First round, no cell dependent GNN
	if training_stage <= 1:
		print ("Pre-training")
		use_recon = False
		higashi_model.only_distance=True
		train(higashi_model,
		      loss=loss,
		      training_data_generator=training_data_generator,
		      validation_data_generator=validation_data_generator,
		      optimizer=[optimizer], epochs=15,
		      load_first=False, save_embed=True)
		pair_ratio = 0.0
		# Training Stage 1
		print ("First stage training")
		higashi_model.only_distance = False
		use_recon = True
		train(higashi_model,
			  loss=loss,
			  training_data_generator=training_data_generator,
			  validation_data_generator=validation_data_generator,
			  optimizer=[optimizer], epochs=embedding_epoch,
			  load_first=False, save_embed=True, save_name="_stage1")
		
		# raise KeyboardInterrupt
		checkpoint = {
			'model_link': higashi_model.state_dict()}

		torch.save(checkpoint, save_path+"_stage1")


	# Loading Stage 1
	checkpoint = torch.load(save_path+"_stage1", map_location=current_device)
	higashi_model.load_state_dict(checkpoint['model_link'])
	node_embedding_init.off_hook([0])

	alpha = 1.0
	beta = 1e-3
	dynamic_pair_ratio = False
	use_recon = False
	pair_ratio = 0.6

	remove_flag = True
	node_embedding2 = GraphSageEncoder_with_weights(features=node_embedding_init, linear_features=node_embedding_init,
													feature_dim=dimensions,
													embed_dim=dimensions,
													num_sample=8, gcn=False, num_list=num_list,
													transfer_range=local_transfer_range, start_end_dict=start_end_dict,
													pass_pseudo_id=False, remove=remove_flag,
													pass_remove=False).to(device)

	higashi_model.encode1.dynamic_nn = node_embedding2
	optimizer = torch.optim.Adam(higashi_model.parameters(), lr=1e-3)


	if impute_no_nbr_flag or impute_with_nbr_flag:

		# Second round, with cell dependent GNN, but no neighbors
		if training_stage <= 2:
			# Training Stage 2
			print("Second stage training")
			train(higashi_model,
				  loss=loss,
				  training_data_generator=training_data_generator,
				  validation_data_generator=validation_data_generator,
				  optimizer=[optimizer], epochs=no_nbr_epoch,
				  load_first=False, save_embed=False)
			checkpoint = {
					'model_link': higashi_model.state_dict()}

			torch.save(checkpoint, save_path + "_stage2")

		# Loading Stage 2
		checkpoint = torch.load(save_path + "_stage2", map_location=current_device)
		higashi_model.load_state_dict(checkpoint['model_link'])

	if training_stage <= 2:
		# Impute Stage 2

		if impute_no_nbr_flag:
			if non_para_impute:
				impute_process(args.config, higashi_model, "%s_nbr_%d_impute"  % (embedding_name, 1), mode, 0, num[0], os.path.join(temp_dir, "sparse_nondiag_adj_nbr_1.npy"))
			else:
				torch.save(higashi_model, save_path + "_stage2_model")
				cell_id_all = np.arange(num[0])
				# print (cell_id_all)
				cell_id_all = np.array_split(cell_id_all, gpu_num-1)
				for i in range(gpu_num-1):
					impute_pool.submit(mp_impute, args.config, save_path + "_stage2_model", "%s_nbr_%d_impute_part_%d" %(embedding_name, 1, i), mode, np.min(cell_id_all[i]), np.max(cell_id_all[i]) + 1, os.path.join(temp_dir, "sparse_nondiag_adj_nbr_1.npy"))
					time.sleep(10)
				impute_pool.shutdown(wait=True)
				linkhdf5("%s_nbr_%d_impute" % (embedding_name, 1), cell_id_all, temp_dir, impute_list)
				# Rank match is to make sure the distribution of predicted values match real population Hi-C values.
				
				impute_pool = ProcessPoolExecutor(max_workers=gpu_num - 1)
				
	if impute_with_nbr_flag:
		train_bce_loss, _, _, _, _ = train_epoch(higashi_model, loss, validation_data_generator, [optimizer])
		update_num_per_eval_epoch = 100
		valid_bce_loss, _, _, _= eval_epoch(higashi_model, loss, validation_data_generator)
		update_num_per_eval_epoch = 10

		print ("train_loss", train_bce_loss, "test_loss", valid_bce_loss)
		
		
		# nbr_mode = 1, the neighbors excludes the cell itself
		# nbr_mode = 0, the neighbors includes the cell itself
		# If the gap between train and valid loss is small, we can include the cell itself in the nbr_list
		# If the gap is large, it indicates an overfitting problem. We would just use the neighboring cells to approximate and add things later
		# Also, if the valid loss is too small, it also indicates overfitting, only use neighbors as well
		nbr_mode = 1 if (train_bce_loss - valid_bce_loss) > 0.1 or valid_bce_loss < 0.1 else 0
		if not impute_no_nbr_flag:
			nbr_mode = 0
		if remove_be_flag:
			nbr_mode = 0
		
		print ("nbr_mode", nbr_mode)
		# Training Stage 3
		print ("getting cell nbr's nbr list")
		
		if remove_be_flag:
			cell_neighbor_list, cell_neighbor_weight_list = get_cell_neighbor_be(nbr_mode)
		else:
			cell_neighbor_list, cell_neighbor_weight_list = get_cell_neighbor(nbr_mode)

		weight_dict = {}
		print (cell_neighbor_list[:10], cell_neighbor_weight_list[:10])
		
		label_info = pickle.load(open(os.path.join(data_dir, "label_info.pickle"), "rb"))
		label = np.array(label_info["cell type"])
		for i in range(len(cell_neighbor_list)-1):
			cell = i+1
			print (label[i], label[cell_neighbor_list[cell]-1])
		
		
		for i in trange(len(cell_neighbor_list)):
			for c, w in zip(cell_neighbor_list[i], cell_neighbor_weight_list[i]):
				weight_dict[(c, i)] = w
		weighted_adj = True
		
		print("processing neighboring info")
		new_sparse_chrom_list = [[] for i in range(len(sparse_chrom_list))]
		for c, chrom in enumerate(chrom_list):
			new_cell_chrom_list = []
			for cell in np.arange(num_list[0]) + 1:
				mtx = 0
				for nbr_cell in cell_neighbor_list[cell]:
					balance_weight = weight_dict[(nbr_cell, cell)]
					mtx = mtx + balance_weight * sparse_chrom_list[c][nbr_cell - 1]
				
				new_cell_chrom_list.append(mtx)
			new_cell_chrom_list = np.array(new_cell_chrom_list)
			new_sparse_chrom_list[c] = new_cell_chrom_list
		new_sparse_chrom_list = np.array(new_sparse_chrom_list)
		
		


		np.save(os.path.join(temp_dir, "weighted_info.npy"), np.array([cell_neighbor_list, weight_dict]), allow_pickle=True)
		# node_embedding2.off_hook()
		# node_embedding1 = GraphSageEncoder_with_weights(features=node_embedding2, linear_features=node_embedding2,
		#                                                 feature_dim=dimensions,
		#                                                 embed_dim=dimensions, node2nbr=neighbor_list,
		#                                                 num_sample=16, gcn=False, num_list=num_list,
		#                                                 transfer_range=0, start_end_dict=start_end_dict,
		#                                                 pass_pseudo_id=True, remove=remove_flag,
		#                                                 pass_remove=False).to(device)
		#
		# higashi_model.encode1.dynamic_nn = node_embedding1

		optimizer = torch.optim.Adam(higashi_model.parameters(), lr=1e-3)

		if training_stage <= 3:
			print("Final stage training")
			train(higashi_model,
				  loss=loss,
				  training_data_generator=training_data_generator,
				  validation_data_generator=validation_data_generator,
				  optimizer=[optimizer], epochs=with_nbr_epoch,  load_first=False)

			checkpoint = {
				'model_link': higashi_model.state_dict()}

			torch.save(checkpoint, save_path+"_stage3")

		# Loading Stage 3
		checkpoint = torch.load(save_path + "_stage3", map_location=current_device)
		higashi_model.load_state_dict(checkpoint['model_link'])
		node_embedding_init.off_hook()
		
		train_bce_loss, _, _, _, _ = train_epoch(higashi_model, loss, validation_data_generator, [optimizer])
		update_num_per_eval_epoch = 100
		valid_bce_loss, _, _, _ = eval_epoch(higashi_model, loss, validation_data_generator)
		update_num_per_eval_epoch = 10
		
		print("train_loss", train_bce_loss, "test_loss", valid_bce_loss)
		
		# raise EOFError

		del train_data, test_data, train_chrom, test_chrom, train_weight, test_weight, training_data_generator, validation_data_generator
		del sparse_chrom_list, weight_dict
		
		# Impute Stage 3
		
		if impute_with_nbr_flag:
			if non_para_impute:
				impute_process(args.config, higashi_model, "%s_nbr_%d_impute"  % (embedding_name, neighbor_num), mode, 0, num[0], os.path.join(temp_dir, "sparse_nondiag_adj_nbr_1.npy" ), os.path.join(temp_dir, "weighted_info.npy"))
			else:
				torch.save(higashi_model, save_path + "_stage3_model")
				cell_id_all = np.arange(num[0])
				cell_id_all = np.array_split(cell_id_all, gpu_num)
				for i in range(gpu_num-1):
					impute_pool.submit(mp_impute, args.config, save_path + "_stage3_model", "%s_nbr_%d_impute_part_%d" %(embedding_name, neighbor_num, i), mode, np.min(cell_id_all[i]), np.max(cell_id_all[i]) + 1, os.path.join(temp_dir, "sparse_nondiag_adj_nbr_1.npy"), os.path.join(temp_dir, "weighted_info.npy"))
					time.sleep(60)
				impute_process(args.config, higashi_model,  "%s_nbr_%d_impute_part_%d" %(embedding_name, neighbor_num, i+1), mode, np.min(cell_id_all[i+1]), np.max(cell_id_all[i+1]) + 1, os.path.join(temp_dir, "sparse_nondiag_adj_nbr_1.npy"), os.path.join(temp_dir, "weighted_info.npy"))
				impute_pool.shutdown(wait=True)
				
				extra_str = "%s_nbr_%d_impute" % (embedding_name, 1) if (impute_no_nbr_flag and nbr_mode == 1 and not remove_be_flag) else None
				
				# When the 1nb imputation is there and nbr_mode=1 (itself is not included during learning), add the predicted values with only 1nb to the neighbor version.
				linkhdf5("%s_nbr_%d_impute" % (embedding_name, neighbor_num), cell_id_all, temp_dir, impute_list, extra_str)


			rank_match_hdf5("%s_nbr_%d_impute" % (embedding_name, neighbor_num), temp_dir, impute_list, config)

		if impute_no_nbr_flag:
			rank_match_hdf5("%s_nbr_%d_impute" % (embedding_name, 1), temp_dir, impute_list, config)