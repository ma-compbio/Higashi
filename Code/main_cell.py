import multiprocessing as mp
import os
import time
import warnings
from Higashi_backend.Modules import *
from Higashi_backend.Functions import *
from Higashi_backend.utils import *
from sklearn.decomposition import PCA
from Impute import impute_process
import argparse
import resource

from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, MinMaxScaler
import pickle
import subprocess

from collections import Counter

def parse_args():
	parser = argparse.ArgumentParser(description="Higashi main program")
	parser.add_argument('-c', '--config', type=str, default="../config_dir/config_ramani.JSON")
	parser.add_argument('-s', '--start', type=int, default=1)
	
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
		return "cuda:%d" % chosen_id
	else:
		return


def forward_batch_hyperedge(model, loss_func, batch_data, batch_weight, y):
	x = batch_data
	w = batch_weight
	pred = model(x)
	
	if use_recon:
		adj = node_embedding_init.embeddings[0](cell_ids)
		targets = node_embedding_init.targets[0](cell_ids)
		_, recon = node_embedding_init.wstack[0](adj, return_recon=True)
		mse_loss = F.mse_loss(recon, targets, reduction="sum") / len(adj)
	else:
		mse_loss = torch.as_tensor([0], dtype=torch.float).to(device)
		
		
	if mode == 'classification':
		main_loss = loss_func(pred, y, weight=w)
		
	elif mode == 'rank':
		pred = F.softplus(pred).float()
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
				makeitzero = F.mse_loss(w[mask_w_eq_zero], pred[mask_w_eq_zero])
				mse_loss += makeitzero
		
	else:
		print ("wrong mode")
		raise EOFError

	domain_loss = torch.as_tensor([0], dtype=torch.float).to(device)
	
	return pred, main_loss, mse_loss, domain_loss


def train_epoch(model, loss_func, training_data_generator, optimizer_list):
	# Epoch operation in training phase
	# Simultaneously train on : hyperedge-prediction (1)
	loss_func = loss_func
	
	model.train()
	
	bce_total_loss = 0
	mse_total_loss = 0
	domain_total_loss = 0
	final_batch_num = 0
	
	batch_num = int(update_num_per_training_epoch / collect_num)
	
	# pool = ProcessPoolExecutor(max_workers=int(cpu_num*1.5))
	pool = ProcessPoolExecutor(max_workers=1)
	p_list = []
	y_list, pred_list = [], []
	
	bar = trange(batch_num * collect_num, desc=' - (Training) ', leave=False, )
	for i in range(batch_num):
		edges_part, edge_weight_part = training_data_generator.next_iter()
		p_list.append(pool.submit(one_thread_generate_neg, edges_part, edge_weight_part, "train_dict"))
	
	for p in as_completed(p_list):
		batch_edge_big, batch_y_big, batch_edge_weight_big = p.result()
		batch_edge_big = np2tensor_hyper(batch_edge_big, dtype=torch.long)
		batch_y_big, batch_edge_weight_big = torch.from_numpy(batch_y_big), torch.from_numpy(batch_edge_weight_big)
		# 	batch_edge_big, batch_y_big, batch_edge_weight_big = one_thread_generate_neg(edges_part, edge_weight_part, "train_dict")
		batch_edge_big, batch_y_big, batch_edge_weight_big = batch_edge_big.to(device), batch_y_big.to(
			device), batch_edge_weight_big.to(device)
		size = int(len(batch_edge_big) / collect_num)
		for j in range(collect_num):
			batch_edge, batch_edge_weight, batch_y = batch_edge_big[j * size: min((j + 1) * size, len(batch_edge_big))], \
			                                         batch_edge_weight_big[
			                                         j * size: min((j + 1) * size, len(batch_edge_big))], \
			                                         batch_y_big[j * size: min((j + 1) * size, len(batch_edge_big))]
			
			pred, loss_bce, loss_mse, loss_domain = forward_batch_hyperedge(model, loss_func, batch_edge,
			                                                                batch_edge_weight, y=batch_y)
			
			y_list.append(batch_y)
			pred_list.append(pred)
			
			final_batch_num += 1
			train_loss = alpha * loss_bce + beta * loss_mse
			# print (train_loss, loss_bce, loss_mse)
			for opt in optimizer_list:
				opt.zero_grad()
			
			# backward
			train_loss.backward()
			
			# update parameters
			for opt in optimizer_list:
				opt.step()
			bar.update(n=1)
			bar.set_description(" - (Training) BCE:  %.3f MSE: %.3f Domain: %.3f Loss: %.3f" %
			                    (loss_bce.item(), loss_mse.item(), loss_domain.item(), train_loss.item()),
			                    refresh=False)
			
			bce_total_loss += loss_bce.item()
			mse_total_loss += loss_mse.item()
			domain_total_loss += loss_domain.item()
		p_list.remove(p)
		del p
	
	y = torch.cat(y_list)
	pred = torch.cat(pred_list)
	auc1, auc2 = roc_auc_cuda(y, pred)
	pool.shutdown(wait=True)
	bar.close()
	return bce_total_loss / final_batch_num, mse_total_loss / final_batch_num, domain_total_loss / final_batch_num, accuracy(
		y, pred), auc1, auc2


def eval_epoch(model, loss_func, validation_data_generator):
	"""Epoch operation in evaluation phase"""
	bce_total_loss = 0
	
	model.eval()
	with torch.no_grad():
		pred, label = [], []
		auc1_list, auc2_list = [], []
		for i in tqdm(range(update_num_per_eval_epoch), desc='  - (Validation)   ', leave=False):
			edges_part, edge_weight_part = validation_data_generator.next_iter()
			
			batch_x, batch_y, batch_w = one_thread_generate_neg(edges_part, edge_weight_part,
			                                                    "test_dict")
			batch_x = np2tensor_hyper(batch_x, dtype=torch.long)
			batch_y, batch_w = torch.from_numpy(batch_y), torch.from_numpy(batch_w)
			batch_x, batch_y, batch_w = batch_x.to(device), batch_y.to(device), batch_w.to(device)
			
			pred_batch, eval_loss, _, _ = forward_batch_hyperedge(model, loss_func, batch_x, batch_w, y=batch_y)
			
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


def generate_negative_cpu(x, dict_type, forward=True):
	global pair_ratio
	rg = np.random.default_rng()
	if dict_type == 'train_dict':
		dict1 = [train_dict, test_dict]
	
	elif dict_type == 'test_dict':
		dict1 = [train_dict, test_dict]
	else:
		dict1 = []
	
	neg_list, new_index = [], []
	
	if forward:
		func1 = pass_
	else:
		func1 = tqdm
	
	# Why neg_num + 1? Because sometimes it may fails to find samples after certain trials.
	# So we just add for trials in the first place
	
	change_list_all = rg.integers(0, x.shape[-1], (len(x), neg_num))
	simple_or_hard_all = rg.random((len(x), neg_num + 1))
	for j, sample in enumerate(func1(x)):
		
		for i in range(neg_num):
			temp = np.copy(sample)
			a = {tuple(temp)}
			change = change_list_all[j, i]
			trial = 0
			
			while any([not a.isdisjoint(d) for d in dict1]):
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
						
						if dict_type == 'train_dict':
							temp[change] = rg.choice(train_cell) + 1
						elif dict_type == 'test_dict':
							temp[change] = rg.choice(test_cell) + 1
						else:
							temp[change] = rg.choice(end - start) + start + 1
				else:
					if dict_type == 'train_dict':
						temp[0] = rg.choice(train_cell) + 1
					elif dict_type == 'test_dict':
						temp[0] = rg.choice(test_cell) + 1
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
				if ((temp[2] - temp[1]) >= max_bin) or (temp[1] == temp[2]) or ((temp[2] - temp[1]) < min_bin):
					temp = np.copy(sample)
				
				if len(neighbor_mask) > 1:
					a = get_neighbor(temp)
				else:
					a = {tuple(temp)}
			
			if len(temp) > 0:
				neg_list.append(temp)
	
	return neg_list


def one_thread_generate_neg(edges_part, edge_weight, dict_type):
	if neg_num == 0:
		# pos_weight = torch.tensor(edge_weight)
		# pos_part = np2tensor_hyper(edges_part, dtype=torch.long)
		y = np.ones((len(edges_part), 1))
		w = np.ones((len(edges_part), 1)) * edge_weight.reshape((-1, 1))
		x = edges_part
	else:
		try:
			neg_list = np.array(generate_negative_cpu(edges_part, dict_type, True))
			# pos_weight = torch.tensor(edge_weight)
			neg_list = neg_list[: len(edges_part) * neg_num, :]
			if len(neg_list) == 0:
				raise EOFError
			# neg = np2tensor_hyper(neg_list, dtype=torch.long)
			# pos_part = np2tensor_hyper(edges_part, dtype=torch.long)
			
			correction = 1.0 if mode == "classification" else 0.0
			y = np.concatenate([np.ones((len(edges_part), 1)),
			                    np.zeros((len(neg_list), 1))])
			w = np.concatenate([np.ones((len(edges_part), 1)) * edge_weight.reshape((-1, 1)),
			                    np.ones((len(neg_list), 1)) * correction])
			x = np.concatenate([edges_part, neg_list])
		except Exception as e:
			print("error from generate neg", e)
			raise EOFError
	
	index = np.random.permutation(len(x))
	x, y, w = x[index], y[index], w[index]
	
	return x, y, w


def train(model, loss, training_data, validation_data, optimizer, epochs, batch_size, load_first, save_embed=False):
	global pair_ratio
	no_improve = 0
	if load_first:
		checkpoint = torch.load(save_path)
		model.load_state_dict(checkpoint['model_link'])
	
	valid_accus = [0]
	train_accus = []
	edges, edge_weight = training_data
	validation_data, validation_weight = validation_data
	
	training_data_generator = DataGenerator(edges, edge_weight, int(batch_size / (neg_num + 1) * collect_num),
	                                              True, num_list)
	validation_data_generator = DataGenerator(validation_data, validation_weight, int(batch_size / (neg_num + 1)),
	                                                False, num_list)
	
	
	for epoch_i in range(epochs):
		if save_embed:
			save_embeddings(model, True)
			save_embeddings(model, False)
		
		print('[ Epoch', epoch_i, 'of', epochs, ']')
		
		start = time.time()
		
		bce_loss, mse_loss, domain_loss, train_accu, auc1, auc2 = train_epoch(
			model, loss, training_data_generator, optimizer)
		print('  - (Training)   bce: {bce_loss: 7.4f}, mse: {mse_loss: 7.4f}, domain: {domain_loss: 7.4f},'
		      ' acc: {accu:3.3f} %, auc: {auc1:3.3f}, aupr: {auc2:3.3f}, '
		      'elapse: {elapse:3.3f} s'.format(
			bce_loss=bce_loss,
			mse_loss=mse_loss,
			domain_loss=domain_loss,
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
		
		checkpoint = {
			'model_link': model.state_dict(),
			'epoch': epoch_i}
		
		# Dynamic pair ratio for stage one
		if dynamic_pair_ratio:
			if pair_ratio < 0.8:
				pair_ratio += 0.1
			elif pair_ratio > 0.8:
				pair_ratio = 0.8
			print("pair_ratio", pair_ratio)
		
		if (not dynamic_pair_ratio) or pair_ratio == 0.8:
			valid_accus += [valid_auc2]
			
			if valid_auc2 >= max(valid_accus):
				print("%.2f to %.2f saving" % (valid_auc2, float(max(valid_accus))))
				torch.save(checkpoint, save_path)
			if len(train_accus) > 0:
				if bce_loss <= (min(train_accus) - 1e-3):
					no_improve = 0
					train_accus += [bce_loss]
				else:
					print(bce_loss, min(train_accus) - 1e-3)
					no_improve += 1
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


def get_neighbor(x):
	result = set()
	a = np.copy(x)
	temp = (a + neighbor_mask)
	temp = np.sort(temp, axis=-1)
	
	for t in temp:
		result.add(tuple(t))
	return result


def save_embeddings(model, origin=False):
	model.eval()
	with torch.no_grad():
		ids = torch.arange(1, num_list[-1] + 1).long().to(device).view(-1)
		embeddings = []
		for j in range(math.ceil(len(ids) / batch_size)):
			x = ids[j * batch_size:min((j + 1) * batch_size, len(ids))]
			if origin:
				embed = node_embedding_init(x)
			else:
				embed = model.node_embedding(x)
			embed = embed.detach().cpu().numpy()
			embeddings.append(embed)
		
		embeddings = np.concatenate(embeddings, axis=0)
		for i in range(len(num_list)):
			start = 0 if i == 0 else num_list[i - 1]
			static = embeddings[int(start):int(num_list[i])]
			
			if origin:
				if i == 0:
					try:
						old_static = np.load(os.path.join(temp_dir, "%s_%d_origin.npy" % (embedding_name, i)))
						update_rate = np.sum((old_static - static) ** 2, axis=-1) / np.sum(old_static ** 2, axis=-1)
						print("update_rate: %f\t%f" % (np.min(update_rate), np.max(update_rate)))
					except Exception as e:
						pass
				np.save(os.path.join(temp_dir, "%s_%d_origin.npy" % (embedding_name, i)), static)
			else:
				np.save(os.path.join(temp_dir, "%s_%d.npy" % (embedding_name, i)), static)
	
	torch.cuda.empty_cache()
	return embeddings


def generate_attributes():
	embeddings = []
	targets = []
	pca_after = []
	
	for c in chrom_list:
		a = np.load(os.path.join(temp_dir, "%s_cell_PCA.npy" % c))
		a = MinMaxScaler((-0.1, 0.1)).fit_transform(a.reshape((-1, 1))).reshape((len(a), -1))
		pca_after.append(a)
	pca_after = np.concatenate(pca_after, axis=-1).astype('float32')
	
	
	
	if coassay:
		cell_attributes = np.load(os.path.join(temp_dir, "pretrain_coassay.npy")).astype('float32')
		targets.append(cell_attributes)
		embeddings.append(cell_attributes)
	else:
		embeddings.append(pca_after)
		targets.append(pca_after)


	
		
	for i, c in enumerate(chrom_list):
		temp = np.load(os.path.join(temp_dir, "%s_bin_adj.npy" % c)).astype('float32')
		temp /= (np.mean(temp, axis=-1, keepdims=True) + 1e-15)
		temp = np.eye(len(temp)).astype('float32')
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
	
	return embeddings, attribute_dict, targets


def reduce_duplicate_normalize_dict(list_of_nb, num=10):
	nb_list = []
	for nb in tqdm(list_of_nb):
		if len(nb) == 0:
			nb_list.append(np.zeros((0, 2)))
		else:
			nb = np.array([[k,nb[k]] for k in nb ])
			
			if len(nb) < num or num < 0:
				nb_list.append(nb)
			else:
				cut_off = np.sort(nb[:, 1])[::-1]
				cut_off = cut_off[num - 1]
				mask = nb[:, 1] >= cut_off
				nb_list.append(nb[mask, :])
	
	return np.array(nb_list)


def get_bin_neighbor_list_dict(data, weight, cell_neighbor_list, cell_neighbor_weight_list, samp_num=10):
	# Feed in data that has already + 1
	print("start getting neighbors")
	weight_dict = {}
	
	for i in trange(len(cell_neighbor_list)):
		for c, w in zip(cell_neighbor_list[i], cell_neighbor_weight_list[i]):
			weight_dict[(i, c)] = w
	
	cell_neighbor_list_inverse = [[] for i in range(num[0] + 1)]
	for i, cell_nbr in enumerate(cell_neighbor_list):
		for c in cell_nbr:
			cell_neighbor_list_inverse[c].append(i)
	
	size = (cell_num + 1) * (int(num_list[-1]) + 1)
	neighbor_list = [Counter() for i in range(size)]
	bulk_neighbor_list = []
	

	for datum, w in tqdm(zip(data, weight), total=len(data)):
		for c in cell_neighbor_list_inverse[datum[0] + 1]:
			balance_weight = weight_dict[(c, datum[0] + 1)]
			neighbor_list[c * (num_list[-1] + 1) + datum[1] + 1][datum[2] + 1] += w * balance_weight
			neighbor_list[c * (num_list[-1] + 1) + datum[2] + 1][datum[1] + 1] +=  w * balance_weight

	
	neighbor_list = reduce_duplicate_normalize_dict(neighbor_list, samp_num)
	
	print(neighbor_list)
	
	return neighbor_list, bulk_neighbor_list

# neighbor_list = [Counter() for i in range(size)]

def get_cell_neighbor(start=1):
	# save_embeddings(higashi_model, True)
	v = np.load(os.path.join(temp_dir, "%s_0_origin.npy" % embedding_name))
	distance = pairwise_distances(v, metric='euclidean')
	distance_sorted = np.sort(distance, axis=-1)
	distance /= np.quantile(distance_sorted[:, 1:neighbor_num].reshape((-1)), q=0.5)
	# distance /= np.mean(distance)
	cell_neighbor_list = [[] for i in range(num[0] + 1)]
	cell_neighbor_weight_list = [[] for i in range(num[0] + 1)]
	for i, d in enumerate(tqdm(distance)):
		neighbor = np.argsort(d)[:neighbor_num]
		weight = np.sort(d)[:neighbor_num]
		
		neighbor_new = neighbor
		new_w = weight
		
		new_w = np.exp(-new_w)
		new_w /= np.sum(new_w)
		# new_w[0] = 0.0
		cell_neighbor_list[i + 1] = (neighbor_new + 1)[start:]
		cell_neighbor_weight_list[i + 1] = (new_w)[start:]
		# cell_neighbor_weight_list[i + 1] /= np.sum(cell_neighbor_weight_list[i + 1])
	
	
	return np.array(cell_neighbor_list), np.array(cell_neighbor_weight_list)


def mp_impute(config_path, path, name, mode):
	cmd = ["python", "Impute.py", config_path, path, name, mode]
	subprocess.call(cmd)


def get_neighbor_mask():
	neighbor_mask = np.zeros((5, 3), dtype='int')
	count = 0
	for i in [-1, 0, 1]:
		for j in [-1, 0, 1]:
			if i !=0 and j !=0:
				continue
			neighbor_mask[count, 1] += i
			neighbor_mask[count, 2] += j
			count += 1
	return neighbor_mask

if __name__ == '__main__':
	
	cpu_num = min(45,int(mp.cpu_count()))
	if torch.cuda.is_available():
		current_device = get_free_gpu()
	else:
		current_device = 'cpu'
		torch.set_num_threads(cpu_num)
		
	global pair_ratio
	
	
	print("cpu_num", cpu_num)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print ("device", device)
	warnings.filterwarnings("ignore")
	rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
	resource.setrlimit(resource.RLIMIT_NOFILE, (3600, rlimit[1]))
	
	impute_pool = ProcessPoolExecutor(max_workers=3)
	
	# Get parameters from config file
	args = parse_args()
	config = get_config(args.config)
	training_stage = args.start
	data_dir = config['data_dir']
	temp_dir = config['temp_dir']
	chrom_list = config['chrom_list']
	print (chrom_list)
	dimensions = config['dimensions']
	impute_list = config['impute_list']
	res = config['resolution']
	neighbor_num = config['neighbor_num']
	local_transfer_range = config['local_transfer_range']
	config_name = config['config_name']
	rank_thres =  config['rank_thres']
	mode = config["loss_mode"]
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
	
	
	#Training related parameters, but these are hard coded as they usually don't require tuning
	# collect_num=x means, one cpu thread would collect x batches of samples
	collect_num = 5
	update_num_per_training_epoch = 1000
	update_num_per_eval_epoch = 10
	
	batch_size = 192
	
	num = np.load(os.path.join(temp_dir, "num.npy"))
	num_list = np.cumsum(num)
	cell_num = num[0]
	cell_ids = (torch.arange(num[0])).long().to(device)
	
	
	# Now start loading data
	data = np.load(os.path.join(temp_dir, "filter_data.npy")).astype('int')
	weight = np.load(os.path.join(temp_dir, "filter_weight.npy")).astype('float32')
	
	
	index = np.arange(len(data))
	np.random.shuffle(index)
	train_index = index[:int(0.85 * len(index))]
	test_index = index[int(0.85 * len(index)):]
	
	
	
	total_possible = 0
	chrom_start_end = np.load(os.path.join(temp_dir, "chrom_start_end.npy"))
	for c in chrom_start_end:
		start, end = c[0], c[1]
		for i in range(start, end):
			for j in range(min(i + min_bin, end), min(end, i + max_bin)):
				total_possible += 1
	
	print(total_possible)
	total_possible *= cell_num
	sparsity = len(data) / total_possible
	print("sparsity", sparsity, len(data), total_possible)
	
	
	neighbor_mask = get_neighbor_mask()
	
	if sparsity > 0.35:
		neg_num = 1
	elif sparsity > 0.25:
		neg_num = 2
	elif sparsity > 0.2:
		neg_num = 3
	elif sparsity > 0.15:
		neg_num = 4
	else:
		neg_num = 5
		
	print("neg_num", neg_num)
	batch_size *= (1 + neg_num)
	
	print("weight", weight, np.min(weight), np.max(weight))
	weight += 1
	# if mode == 'rank':
	# 	weight = KBinsDiscretizer(n_bins=50, encode='ordinal', strategy='quantile').fit_transform(
	# 		weight.reshape((-1, 1))).reshape((-1)) + 2
	print("weight", weight, np.min(weight), np.max(weight))
	# print ("weight distribution")
	# for w in np.unique(weight):
	# 	print (w, np.sum(weight == w))
	
	train_data = data[train_index]
	train_weight = weight[train_index]
	test_data = data[test_index]
	test_weight = weight[test_index]
	
	
	train_mask = ((train_data[:, 2] - train_data[:, 1]) >= min_bin) & ((train_data[:, 2] - train_data[:, 1]) < max_bin)
	train_data = train_data[train_mask]
	train_weight = train_weight[train_mask]
	
	test_mask = ((test_data[:, 2] - test_data[:, 1]) >= min_bin) & ((test_data[:, 2] - test_data[:, 1]) < max_bin)
	test_data = test_data[test_mask]
	test_weight = test_weight[test_mask]
	
	
	print("Node type num", num, num_list)
	start_end_dict = np.load(os.path.join(temp_dir, "start_end_dict.npy"))
	start_end_dict = np.concatenate([np.zeros((1, 2)), start_end_dict], axis=0).astype('int')
	
	
	print("start_end_dict", start_end_dict.shape, start_end_dict)
	print("data", data, np.max(data), data.shape)
	
	
	train_cell = np.unique(train_data[:, 0])
	test_cell = np.unique(test_data[:, 0])
	
	print(train_data, test_data)
	try:
		cell_feats = np.load(os.path.join(temp_dir, "cell_feats.npy")).astype('float32')
		cell_feats = np.concatenate([np.zeros((1, cell_feats.shape[-1])), cell_feats], axis=0).astype('float32')
	except:
		cell_feats = None
	print ("cell_feats", cell_feats)
	embeddings_initial, attribute_dict, targets_initial = generate_attributes()
	
	# Add 1 for the padding index
	print("adding pad idx")
	train_data = add_padding_idx(train_data)
	test_data = add_padding_idx(test_data)
	
	
	
	compress = False
	initial_set = set()
	if neg_num > 0:
		train_dict = parallel_build_hash(train_data, "build_hash", num, initial=initial_set, compress=compress)
		test_dict = parallel_build_hash(test_data, "build_hash", num, initial=initial_set, compress=compress)
	else:
		train_dict, test_dict = set(), set()
		
	print("dict_size", len(train_dict), len(test_dict))
	
	
	print("train_weight", train_weight)
	
	if mode == 'classification':
		train_weight_mean = np.mean(train_weight)
		
		train_weight, test_weight = transform_weight_class(train_weight, train_weight_mean, neg_num), \
		                            transform_weight_class(test_weight, train_weight_mean, neg_num)
	
	print(train_weight, np.min(train_weight), np.max(train_weight))
	print("train data amount", len(train_data))
	
	
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
		node_embedding=node_embedding_init,
		diag_mask=True,
		bottle_neck=dimensions,
		attribute_dict=attribute_dict,
		cell_feats=cell_feats,
		encoder_dynamic_nn=node_embedding_init,
		encoder_static_nn=node_embedding_init).to(device)
	
	loss = F.binary_cross_entropy_with_logits
	
	for name, param in higashi_model.named_parameters():
		print(name, param.requires_grad, param.shape)
	optimizer = torch.optim.Adam(higashi_model.parameters(),
	                              lr=1e-3)
	
	
	model_parameters = filter(lambda p: p.requires_grad, higashi_model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print("params to be trained", params)
	alpha = 1.0
	beta = 1e-2
	dynamic_pair_ratio = True
	use_recon = True
	pair_ratio = 0.0
	cell_neighbor_list = [[i] for i in range(num[0] + 1)]
	cell_neighbor_weight_list = [[1] for i in range(num[0] + 1)]
	
	
	# First round, no cell dependent GNN
	if training_stage <= 1:
		# Training Stage 1
		train(higashi_model,
		      loss=loss,
		      training_data=(train_data, train_weight),
		      validation_data=(test_data, test_weight),
		      optimizer=[optimizer], epochs=120, batch_size=batch_size,
		      load_first=False, save_embed=True)
		# raise KeyboardInterrupt
		checkpoint = {
			'model_link': higashi_model.state_dict()}

		torch.save(checkpoint, save_path+"_stage1")

	
	# Loading Stage 1
	checkpoint = torch.load(save_path+"_stage1", map_location=current_device)
	higashi_model.load_state_dict(checkpoint['model_link'])
	node_embedding_init.off_hook([0])
	
	# if training_stage <= 1:
	# 	# Impute Stage 1
	# 	torch.save(higashi_model, save_path+"_stage1_model")
	# 	# impute_pool.submit(mp_impute, args.config, save_path+"_stage1_model", embedding_name + "_all", mode)
	
	original_data = np.load(os.path.join(temp_dir, "filter_data.npy")).astype('int')
	original_weight = np.load(os.path.join(temp_dir, "filter_weight.npy")).astype('float32')
	mask = ((original_data[:, 2] - original_data[:, 1]) < max_bin) & ((original_data[:, 2] - original_data[:, 1]) >= min_bin)
	original_data = original_data[mask]
	original_weight = original_weight[mask]
	original_weight = np.log10(original_weight+1)
	print ("GCN weight", original_weight, np.min(original_weight), np.max(original_weight))
	neighbor_list, bulk_neighbor_list = get_bin_neighbor_list_dict(original_data, original_weight,
		                                                               cell_neighbor_list, cell_neighbor_weight_list, 32)
	
	alpha = 1.0
	beta = 1e-2
	dynamic_pair_ratio = False
	use_recon = False
	pair_ratio = 0.6
	
	remove_flag = True
	node_embedding2 = GraphSageEncoder_with_weights(features=node_embedding_init, linear_features=node_embedding_init,
	                                                feature_dim=dimensions,
	                                                embed_dim=dimensions, node2nbr=neighbor_list,
	                                                num_sample=8, gcn=False, num_list=num_list,
	                                                transfer_range=local_transfer_range, start_end_dict=start_end_dict,
	                                                pass_pseudo_id=False, remove=remove_flag,
	                                                pass_remove=False).to(device)
	
	higashi_model.encode1.dynamic_nn = node_embedding2
	optimizer = torch.optim.AdamW(higashi_model.parameters(), lr=1e-3, weight_decay=0.01)
	
	# Second round, with cell dependent GNN, but no neighbors
	if training_stage <= 2:
		# Training Stage 2
		train(higashi_model,
		      loss=loss,
		      training_data=(train_data, train_weight),
		      validation_data=(test_data, test_weight),
		      optimizer=[optimizer], epochs=60, batch_size=batch_size,
		      load_first=False, save_embed=False)
		checkpoint = {
				'model_link': higashi_model.state_dict()}
	
		torch.save(checkpoint, save_path + "_stage2")
	
	if training_stage <= 2:
		# Impute Stage 2
		torch.save(higashi_model, save_path + "_stage2_model")
		impute_pool.submit(mp_impute, args.config, save_path + "_stage2_model", "%s_nbr_%d_impute_1l" %(embedding_name, 1), mode)
	
	
	# Loading Stage 2
	checkpoint = torch.load(save_path + "_stage2", map_location=current_device)
	higashi_model.load_state_dict(checkpoint['model_link'])
	
	validation_data_generator = DataGenerator(test_data, test_weight, int(batch_size / (neg_num + 1)),
	                                                False, num_list)
	train_bce_loss, _, _, _, _, _ = train_epoch(higashi_model, loss, validation_data_generator, [optimizer])
	valid_bce_loss, _, _, _= eval_epoch(higashi_model, loss, validation_data_generator)
	
	print ("train_loss", train_bce_loss, "test_loss", valid_bce_loss)
	
	# If the gap between train and valid loss is small, we can include the cell itself in the nbr_list
	# If the gap is large, it indicates an overfitting problem. We would just use the neiboring cells to approximate
	nbr_mode = 0 if (train_bce_loss - valid_bce_loss) < 0.1 or valid_bce_loss > 0.1 else 1
	# nbr_mode = 0
	print ("nbr_mode", nbr_mode)
	# Training Stage 3
	print ("getting cell nbr's nbr list")
	cell_neighbor_list, cell_neighbor_weight_list = get_cell_neighbor(nbr_mode)
	# cell_neighbor_weight_list = [[1 / neighbor_num] * 5 for i in range(num[0] + 1)]
	print("nbr_list",cell_neighbor_list[:10], "nbr_weight_list", cell_neighbor_weight_list[:10])
	del neighbor_list, bulk_neighbor_list
	neighbor_list, bulk_neighbor_list = get_bin_neighbor_list_dict(original_data, original_weight, cell_neighbor_list,
	                                                               cell_neighbor_weight_list, 32)
	
	node_embedding2.node2nbr = neighbor_list
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
	
	optimizer = torch.optim.AdamW(higashi_model.parameters(), lr=1e-3, weight_decay=0.01)
	
	if training_stage <= 3:
		train(higashi_model,
		      loss=loss,
		      training_data=(train_data, train_weight),
		      validation_data=(test_data, test_weight),
		      optimizer=[optimizer], epochs=45, batch_size=batch_size, load_first=False)
	
		checkpoint = {
			'model_link': higashi_model.state_dict()}
	
		torch.save(checkpoint, save_path+"_stage3")
	
	# Loading Stage 3
	checkpoint = torch.load(save_path + "_stage3", map_location=current_device)
	higashi_model.load_state_dict(checkpoint['model_link'])
	
	train(higashi_model,
	      loss=loss,
	      training_data=(train_data, train_weight),
	      validation_data=(test_data, test_weight),
	      optimizer=[optimizer], epochs=0, batch_size=batch_size,
	      load_first=False, save_embed=True)
	
	# Impute Stage 3
	impute_process(args.config, higashi_model, "%s_nbr_%d_impute" %(embedding_name, neighbor_num), mode)
	impute_pool.shutdown(wait=True)