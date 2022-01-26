import copy
import random
import math

import numpy as np
import torch.cuda

from .utils import *
from .Functions import *
import multiprocessing
import time
from torch.nn.utils.rnn import pad_sequence
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.sparse import diags, vstack
from scipy.stats import norm

try:
	get_ipython()
	from tqdm.notebook import tqdm, trange
except:
	pass

cpu_num = multiprocessing.cpu_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
activation_func = swish

# Code adapted from scVI
def log_zinb_positive(
	x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, pi: torch.Tensor, eps=1e-8
):
	"""
	Log likelihood (scalar) of a minibatch according to a zinb model.

	Parameters
	----------
	x
		Data
	mu
		mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
	theta
		inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
	pi
		logit of the dropout parameter (real support) (shape: minibatch x vars)
	eps
		numerical stability constant

	Notes
	-----
	We parametrize the bernoulli using the logits, hence the softplus functions appearing.
	"""
	# theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless of batch or labels)
	# if theta.ndimension() == 1:
	#     theta = theta.view(
	#         1, theta.size(0)
	#     )  # In this case, we reshape theta for broadcasting
	
	
	softplus_pi = F.softplus(-pi)  #  uses log(sigmoid(x)) = -softplus(-x)
	log_theta_eps = torch.log(theta + eps)
	log_theta_mu_eps = torch.log(theta + mu + eps)
	pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

	case_zero = F.softplus(pi_theta_log) - softplus_pi
	mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

	case_non_zero = (
		-softplus_pi
		+ pi_theta_log
		+ x * (torch.log(mu + eps) - log_theta_mu_eps)
		+ torch.lgamma(x + theta)
		- torch.lgamma(theta)
		- torch.lgamma(x + 1)
	)
	mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

	res = mul_case_zero + mul_case_non_zero
	
	return res


class Wrap_Embedding(torch.nn.Embedding):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	
	def forward(self, *input):
		return super().forward(*input)
	
	def features(self, *input):
		return self.forward(*input)
	
	def start_fix(self):
		return
	
	def fix_cell(self, cell_list=None, bin_id=None):
		return
	

# Used only for really big adjacency matrix
class SparseEmbedding(nn.Module):
	def __init__(self, embedding_weight, sparse=False, cpu=False):
		super().__init__()
		# print("Initializing embedding, shape", embedding_weight.shape)
		self.sparse = sparse
		self.cpu_flag = cpu
		
		if self.cpu_flag:
			print("CPU mode")
			self_device = "cpu"
		else:
			self_device = device
		if self.sparse:
			print ("Sparse mode")
			self.embedding = embedding_weight
		else:
			if type(embedding_weight) is torch.Tensor:
				self.embedding = embedding_weight.to(self_device)
			elif type(embedding_weight) is np.ndarray:
				try:
					self.embedding = torch.from_numpy(
						np.array(embedding_weight.todense())).to(self_device)
				except BaseException:
					self.embedding = torch.from_numpy(
						np.array(embedding_weight)).to(self_device)
			else:
				print("Sparse Embedding Error", type(embedding_weight))
				self.sparse = True
				self.embedding = embedding_weight
	
	def forward(self, x):
		
		if self.sparse:
			x = x.cpu().numpy()
			x = x.reshape((-1))
			temp = np.asarray((self.embedding[x, :]).todense())
			return torch.from_numpy(temp).to(device, non_blocking=True)
		if self.cpu:
			temp = self.embedding[x.cpu(), :]
			return temp.to(device, non_blocking=True)
		else:
			return self.embedding[x, :]
	
	
# Deep Auto-encoder with tied or partial tied weights (reduce the number of parameters to be trained)
class TiedAutoEncoder(nn.Module):
	def __init__(self, shape_list: list,
				 use_bias=True,
				 tied_list=None,
				 add_activation=False,
				 dropout=None,
				 layer_norm=False,
				 activation=None):
		
		super().__init__()
		if tied_list is None:
			tied_list = []
		self.add_activation = add_activation
		self.weight_list = []
		self.reverse_weight_list = []
		self.bias_list = []
		self.use_bias = use_bias
		self.recon_bias_list = []
		self.shape_list = shape_list
		self.activation = activation
		if self.activation is None:
			self.activation = activation_func
		# Generating weights for the tied autoencoder
		for i in range(len(shape_list) - 1):
			p = nn.parameter.Parameter(torch.FloatTensor(shape_list[i + 1], shape_list[i]).to(device, non_blocking=True))
			self.weight_list.append(p)
			if i not in tied_list:
				self.reverse_weight_list.append(
					nn.parameter.Parameter(torch.FloatTensor(shape_list[i + 1], shape_list[i]).to(device, non_blocking=True)))
			else:
				self.reverse_weight_list.append(p)
			
			self.bias_list.append(nn.parameter.Parameter(torch.FloatTensor(shape_list[i + 1]).to(device, non_blocking=True)))
			self.recon_bias_list.append(nn.parameter.Parameter(torch.FloatTensor(shape_list[i]).to(device, non_blocking=True)))
		
		# reverse the order of the decoder.
		self.recon_bias_list = self.recon_bias_list[::-1]
		self.reverse_weight_list = self.reverse_weight_list[::-1]
		
		self.weight_list = nn.ParameterList(self.weight_list)
		self.reverse_weight_list = nn.ParameterList(self.reverse_weight_list)
		self.bias_list = nn.ParameterList(self.bias_list)
		self.recon_bias_list = nn.ParameterList(self.recon_bias_list)
		
		# Initialize the parameters
		self.reset_parameters()
		
		if dropout is not None:
			self.dropout = nn.Dropout(dropout)
		else:
			self.dropout = None
			
		if layer_norm:
			self.layer_norm = nn.LayerNorm(shape_list[-1])
		else:
			self.layer_norm = None
		self.tied_list = tied_list
		self.input_dropout = nn.Dropout(0.1)
	
	def reset_parameters(self):
		for i, w in enumerate(self.weight_list):
			nn.init.kaiming_uniform_(self.weight_list[i], a=0.0, mode='fan_in', nonlinearity='leaky_relu')
			nn.init.kaiming_uniform_(self.reverse_weight_list[i], a=0.0, mode='fan_out', nonlinearity='leaky_relu')
			
		for i, b in enumerate(self.bias_list):
			fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight_list[i])
			bound = 1 / math.sqrt(fan_in)
			torch.nn.init.uniform_(self.bias_list[i], -bound, bound)
		
		temp_weight_list = self.weight_list[::-1]
		for i, b in enumerate(self.recon_bias_list):
			fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(temp_weight_list[i])
			bound = 1 / math.sqrt(fan_out)
			torch.nn.init.uniform_(self.recon_bias_list[i], -bound, bound)
	
	def untie(self):
		new_reverse_weight_list = []
		
		for w in self.reverse_weight_list:
			new_reverse_weight_list.append(nn.parameter.Parameter(torch.ones_like(w).to(device, non_blocking=True)))
		for i in range(len(new_reverse_weight_list)):
			nn.init.kaiming_uniform_(new_reverse_weight_list[i], a=0.0, mode='fan_out', nonlinearity='leaky_relu')
		self.reverse_weight_list = nn.ParameterList(new_reverse_weight_list)
		
		for i, b in enumerate(self.recon_bias_list):
			fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.reverse_weight_list[i])
			bound = 1 / math.sqrt(fan_out)
			torch.nn.init.uniform_(self.recon_bias_list[i], -bound, bound)
	
	def encoder(self, input):
		encoded_feats = input
		for i in range(len(self.weight_list)):
			if self.use_bias:
				encoded_feats = F.linear(encoded_feats, self.weight_list[i], self.bias_list[i])
			else:
				encoded_feats = F.linear(encoded_feats, self.weight_list[i])
			
			if i < len(self.weight_list) - 1:
				encoded_feats = self.activation(encoded_feats)
				if self.dropout is not None:
					encoded_feats = self.dropout(encoded_feats)
		
		if self.layer_norm is not None:
			encoded_feats = self.layer_norm(encoded_feats)
		
		if self.add_activation:
			encoded_feats = self.activation(encoded_feats)
		return encoded_feats
	
	def decoder(self, encoded_feats):
		if self.add_activation:
			reconstructed_output = encoded_feats
		else:
			reconstructed_output = self.activation(encoded_feats)
		
		reverse_weight_list = self.reverse_weight_list
		recon_bias_list = self.recon_bias_list
		
		for i in range(len(reverse_weight_list)):
			reconstructed_output = F.linear(reconstructed_output, reverse_weight_list[i].t(),
											recon_bias_list[i])
			
			if i < len(recon_bias_list) - 1:
				reconstructed_output = self.activation(reconstructed_output)
		return reconstructed_output
	
	def forward(self, input, return_recon=False):
		
		encoded_feats = self.encoder(input)
		if return_recon:
			if not self.add_activation:
				reconstructed_output = self.activation(encoded_feats)
			else:
				reconstructed_output = encoded_feats
			
			if self.dropout is not None:
				reconstructed_output = self.dropout(reconstructed_output)
			
			reconstructed_output = self.decoder(reconstructed_output)
			
			return encoded_feats, reconstructed_output
		else:
			return encoded_feats
	
	def fit(self, data: np.ndarray,
			epochs=10, sparse=True, sparse_rate=None, classifier=False, early_stop=True, batch_size=-1, targets=None):
		
		if self.shape_list[1] < data.shape[1]:
			pca = PCA(n_components=self.shape_list[1]).fit(data)
			self.weight_list[0].data = torch.from_numpy(pca.components_).float().to(device, non_blocking=True)
			self.reverse_weight_list[-1].data = torch.from_numpy(pca.components_).float().to(device, non_blocking=True)
			
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		data = torch.from_numpy(data).to(device, non_blocking=True)
		
		
		
		if batch_size < 0:
			batch_size = int(len(data))
		bar = trange(epochs, desc="")
		
		no_improve_count = 0
		for i in bar:
			batch_index = torch.randint(0, int(len(data)), (batch_size,)).to(device, non_blocking=True)
			encode, recon = self.forward(data[batch_index], return_recon=True)
			optimizer.zero_grad()
			
			if sparse:
				loss = sparse_autoencoder_error(recon, targets[batch_index], sparse_rate)
			elif classifier:
				loss = F.binary_cross_entropy_with_logits(recon, (targets[batch_index] > 0).float())
			else:
				loss = F.mse_loss(recon, targets[batch_index])  # / len(recon)
			
			if i == 0:
				loss_best = float(loss.item())
			
			loss.backward()
			optimizer.step()
			
			if early_stop:
				if i >= 50:
					if loss.item() < loss_best * 0.99:
						loss_best = loss.item()
						no_improve_count = 0
					else:
						no_improve_count += 1
					if no_improve_count >= 30:
						break
			bar.set_description("%.3f" % (loss.item()), refresh=False)
		if epochs > 0:
			print("loss", loss.item(), "loss best", loss_best, "epochs", i)
			print()
		torch.cuda.empty_cache()
	
	def predict(self, data):
		self.eval()
		data = torch.from_numpy(data).to(device, non_blocking=True)
		with torch.no_grad():
			encode = self.forward(data)
		self.train()
		torch.cuda.empty_cache()
		return encode.cpu().detach().numpy()


# Deep Auto-encoder
class AutoEncoder(nn.Module):
	def __init__(self, encoder_shape_list, decoder_shape_list,
				 use_bias=True,
				 add_activation=False,
				 dropout=None,
				 layer_norm=False):
		
		super().__init__()
		self.add_activation = add_activation
		self.weight_list = []
		self.reverse_weight_list = []
		self.use_bias = use_bias
		
		# Generating weights for the tied autoencoder
		for i in range(len(encoder_shape_list) - 1):
			self.weight_list.append(nn.Linear(encoder_shape_list[i], encoder_shape_list[i+1]).to(device, non_blocking=True))
		for i in range(len(decoder_shape_list) - 1):
			self.reverse_weight_list.append(nn.Linear(decoder_shape_list[i], decoder_shape_list[i+1]).to(device, non_blocking=True))
			
		self.reverse_weight_list = nn.ModuleList(self.reverse_weight_list)
		self.weight_list = nn.ModuleList(self.weight_list)
		
		if dropout is not None:
			self.dropout = nn.Dropout(dropout)
		else:
			self.dropout = None
		
		if layer_norm:
			self.layer_norm_stack = []
			for i in range(len(encoder_shape_list) - 1):
				self.layer_norm_stack.append(nn.LayerNorm(encoder_shape_list[i+1]).to(device, non_blocking=True))
			
		else:
			self.layer_norm_stack = None
	
	
	
	def encoder(self, input):
		encoded_feats = input
		for i in range(len(self.weight_list)):
			encoded_feats = self.weight_list[i](encoded_feats)
			
			if i < len(self.weight_list) - 1:
				encoded_feats = activation_func(encoded_feats)
				if self.dropout is not None:
					encoded_feats = self.dropout(encoded_feats)
		
			if self.layer_norm_stack is not None:
				encoded_feats = self.layer_norm_stack[i](encoded_feats)
		
		if self.add_activation:
			encoded_feats = activation_func(encoded_feats)
		return encoded_feats
	
	def decoder(self, encoded_feats):
		if self.add_activation:
			reconstructed_output = encoded_feats
		else:
			reconstructed_output = activation_func(encoded_feats)
		
		reverse_weight_list = self.reverse_weight_list
		
		for i in range(len(reverse_weight_list)):
			reconstructed_output = reverse_weight_list[i](reconstructed_output)
			
			if i < len(reverse_weight_list) - 1:
				reconstructed_output = activation_func(reconstructed_output)
		return reconstructed_output
	
	def forward(self, input, return_recon=False):
		
		encoded_feats = self.encoder(input)
		if return_recon:
			reconstructed_output = encoded_feats
			
			if self.dropout is not None:
				reconstructed_output = self.dropout(reconstructed_output)
			
			reconstructed_output = self.decoder(reconstructed_output)
			
			return encoded_feats, reconstructed_output
		else:
			return encoded_feats
	
	def fit(self, data, epochs=10, sparse=True, sparse_rate=None, classifier=False, early_stop=True, batch_size=-1, targets=None):
		optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
		data = torch.from_numpy(data).to(device, non_blocking=True)
		
		if batch_size < 0:
			batch_size = len(data)
		bar = trange(epochs, desc="")
		
		
		if targets is None:
			targets=data
		
		no_improve_count = 0
		for i in bar:
			batch_index = torch.randint(0, len(data), (batch_size,)).to(device, non_blocking=True)
			encode, recon = self.forward(data[batch_index], return_recon=True)
			optimizer.zero_grad()
			
			
			
			if sparse:
				loss = sparse_autoencoder_error(recon, targets[batch_index], sparse_rate)
			elif classifier:
				loss = F.binary_cross_entropy_with_logits(recon, (targets[batch_index] > 0).float())
			else:
				loss = F.mse_loss(recon, targets[batch_index], reduction="sum") / len(batch_index)
			
			if i == 0:
				loss_best = float(loss.item())
				
				
			loss.backward()
			optimizer.step()
			
			if early_stop:
				if i >= 50:
					if loss.item() < loss_best * 0.99:
						loss_best = loss.item()
						no_improve_count = 0
					else:
						no_improve_count += 1
					if no_improve_count >= 50:
						break
			
			bar.set_description("%.3f" % (loss.item()), refresh=False)
		
		print("loss", loss.item(), "loss best", loss_best, "epochs", i)
		print()
		torch.cuda.empty_cache()
	
	def predict(self, data):
		self.eval()
		data = torch.from_numpy(data).to(device, non_blocking=True)
		with torch.no_grad():
			encode = self.forward(data)
		self.train()
		torch.cuda.empty_cache()
		return encode.cpu().detach().numpy()


# Multiple Embedding is a module that passes nodes to different branch of neural network to generate embeddings
# The neural network to use would be dependent to the node ids (the input num_list parameters)
# If the num_list is [0, 1000, 2000,...,]
# Then node 0~1000 would pass through NN1, 1000~200 would pass through NN2...
# target weights represent the auxilary task that the embedding would do.
class MultipleEmbedding(nn.Module):
	def __init__(self, embedding_weights, dim, sparse=True, num_list=None, target_weights=None):
		
		super().__init__()
		
		if target_weights is None:
			target_weights = embedding_weights
		
		self.dim = dim
		self.num_list = torch.tensor([0] + list(num_list)).to(device, non_blocking=True)
		
		# searchsort_table is a fast mapping between node id and the neural network to use for generate embeddings
		self.searchsort_table = torch.zeros(num_list[-1] + 1).long().to(device, non_blocking=True)
		for i in range(len(self.num_list) - 1):
			self.searchsort_table[self.num_list[i] + 1:self.num_list[i + 1] + 1] = i
		self.searchsort_table_one_hot = torch.zeros([len(self.searchsort_table), self.searchsort_table.max() + 1])
		x = torch.range(0, len(self.searchsort_table) - 1, dtype=torch.long)
		self.searchsort_table_one_hot[x, self.searchsort_table] = 1
		self.searchsort_table = self.searchsort_table_one_hot
		self.searchsort_table[0] = 0
		self.searchsort_table = self.searchsort_table.bool().to(device, non_blocking=True)
		
		
		self.embeddings = []
		complex_flag = False
		for i, w in enumerate(embedding_weights):
			self.embeddings.append(SparseEmbedding(w, sparse))

		
		self.targets = []
		complex_flag = False
		for i, w in enumerate(target_weights):
			self.targets.append(SparseEmbedding(w, sparse))
		
		
		# Generate a test id to test the output size of each embedding modules.
		test = torch.zeros(1, device=device).long()
		self.input_size = []
		for w in self.embeddings:
			result = w(test)
			if type(result) == tuple:
				result = result[0]
			self.input_size.append(result.shape[-1])
		
		self.layer_norm = nn.LayerNorm(self.dim).to(device, non_blocking=True)
		
		self.wstack = []
		
		i = 0
		if self.input_size[i] == target_weights[i].shape[-1]:
			self.wstack.append(
				TiedAutoEncoder([self.input_size[i], self.dim], add_activation=False, tied_list=[]))
		else:
			self.wstack.append(AutoEncoder([self.input_size[i], self.dim], [self.dim, target_weights[i].shape[-1]],
										   add_activation=True))
		
		for i in range(1, len(self.embeddings)):
			self.wstack.append(TiedAutoEncoder([self.input_size[i], self.dim],add_activation=True, tied_list=[]))
			
		
		self.wstack = nn.ModuleList(self.wstack)
		self.on_hook_embedding = nn.ModuleList([nn.Sequential(w,
														 self.wstack[i]
														 ) for i, w in enumerate(self.embeddings)])
		self.on_hook_set = set([i for i in range(len(self.embeddings))])
		self.off_hook_embedding = [i for i in range(len(self.embeddings))]
		self.features = self.forward
		
	# route_nn means for this batch of nodes use the xth NN to get embeddings
	def forward(self, x, *args, route_nn=None):
		if len(x.shape) > 1:
			sz_b, len_seq = x.shape
			x = x.view(-1)
			reshape_flag = True
		else:
			reshape_flag = False
		
		if route_nn is None:
			final = torch.zeros((len(x), self.dim), device=device).float()
			# ind is a bool type array
			ind = self.searchsort_table[x]
			node_type = torch.nonzero(torch.any(ind, dim=0)).view(-1)
	
			for i in node_type:
				mask = ind[:, i]
				if int(i) in self.on_hook_set:
						final[mask] = self.on_hook_embedding[i](x[mask] - self.num_list[i] - 1)
	
				else:
					final[mask] = self.off_hook_embedding[i](x[mask] - self.num_list[i] - 1)
		else:
			i = route_nn
			if int(i) in self.on_hook_set:
				final = self.on_hook_embedding[i](x - self.num_list[i] - 1)
			
			else:
				final = self.off_hook_embedding[i](x - self.num_list[i] - 1)
		
		if reshape_flag:
			final = final.view(sz_b, len_seq, -1)
		
		return final
	
	# No longer do BP through a list of embedding modules.
	def off_hook(self, off_hook_list=[]):
		if len(off_hook_list) == 0:
			off_hook_list = list(range(len(self.wstack)))
		for index in off_hook_list:
			ae = self.wstack[index]
			for w in ae.weight_list:
				w.requires_grad = False
			for w in ae.reverse_weight_list:
				w.requires_grad = False
			for b in ae.bias_list:
				b.requires_grad = False
			for b in ae.recon_bias_list:
				b.requires_grad = False
			
			ids = torch.arange(start=0, end=self.num_list[index + 1] - self.num_list[index], device=device)
			with torch.no_grad():
				embed = self.on_hook_embedding[index](ids).detach()
			self.embeddings[index] = self.embeddings[index].cpu()
			try:
				self.targets[index] = self.targets[index].cpu()
			except:
				pass
			self.off_hook_embedding[index] = SparseEmbedding(embed, False)
			try:
				self.on_hook_set.remove(index)
			except:
				pass
	
	def on_hook(self, on_hook_list=[]):
		if len(on_hook_list) == 0:
			on_hook_list = list(range(len(self.wstack)))
		for index in on_hook_list:
			ae = self.wstack[index]
			for w in ae.weight_list:
				w.requires_grad = True
			for w in ae.reverse_weight_list:
				w.requires_grad = True
			for b in ae.bias_list:
				b.requires_grad = True
			for b in ae.recon_bias_list:
				b.requires_grad = True
			self.embeddings[index] = self.embeddings[index].to(device, non_blocking=True)
			self.targets[index] = self.targets[index].to(device, non_blocking=True)
			self.on_hook_set.add(index)
	
	def start_fix(self):
		return
	
	def fix_cell(self, cell=None, bin_id=None):
		return


class Hyper_SAGNN(nn.Module):
	def __init__(
			self,
			n_head,
			d_model,
			d_k,
			d_v,
			diag_mask,
			bottle_neck,
			attribute_dict=None,
			cell_feats=None,
			encoder_dynamic_nn=None,
			encoder_static_nn=None,
			chrom_num=1):
		super().__init__()
		
		self.pff_classifier = PositionwiseFeedForward(
			[d_model, int(d_model / 2), 1])
		self.pff_classifier_var = PositionwiseFeedForward(
			[d_model, int(d_model / 2), 1])
		self.pff_classifier_proba = PositionwiseFeedForward(
			[d_model, int(d_model / 2), 1])
		self.encode_list = []
		self.encode1 = EncoderLayer(
				n_head,
				d_model,
				d_k,
				d_v,
				dropout_mul=0.3,
				dropout_pff=0.4,
				diag_mask=diag_mask,
				bottle_neck=bottle_neck,
				dynamic_nn=encoder_dynamic_nn,
				static_nn=encoder_static_nn)
		
		self.diag_mask_flag = diag_mask
		
		self.layer_norm1 = nn.LayerNorm(d_model)
		self.layer_norm2 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(0.3)
		if attribute_dict is not None:
			self.attribute_dict = torch.from_numpy(attribute_dict).to(device, non_blocking=True)
			input_size = self.attribute_dict.shape[-1] * 2 + cell_feats.shape[-1]
			self.extra_proba = FeedForward([input_size, 4, 1])
			self.extra_proba2 = FeedForward([input_size, 4, 1])
			self.extra_proba3 = FeedForward([input_size, 4, 1])
			self.attribute_dict_embedding = nn.Embedding(len(self.attribute_dict), 1, padding_idx=0)
			self.attribute_dict_embedding.weight = nn.Parameter(self.attribute_dict)
			self.attribute_dict_embedding.weight.requires_grad = False
			self.cell_feats = torch.from_numpy(cell_feats).to(device, non_blocking=True)
		self.only_distance = False
		self.only_model = False
		self.chrom_num = chrom_num
		self.d_model = d_model
	
	def get_embedding(self, x, x_chrom, slf_attn_mask=None, non_pad_mask=None, chroms_in_batch=None):
		# if slf_attn_mask is None:
		# 	slf_attn_mask = get_attn_key_pad_mask(seq_k=x, seq_q=x)
		# 	non_pad_mask = get_non_pad_mask(x)
		

		dynamic, static = self.encode1(x, x, x_chrom, slf_attn_mask, non_pad_mask, chroms_in_batch=chroms_in_batch)
		
		if torch.sum(torch.isnan(dynamic)) > 0:
			print ("nan error", x, dynamic, static)
			raise EOFError
			
		return dynamic, static
	
	def forward(self, x, x_chrom, mask=None, chroms_in_batch=None):
		x = x.long()
		sz_b, len_seq = x.shape
		if self.attribute_dict is not None:
			if not self.only_model:
				distance = torch.cat([self.attribute_dict_embedding(x[:, 1]), self.attribute_dict_embedding(x[:, 2]), self.cell_feats[x[:, 0]]], dim=-1)
				distance_proba = self.extra_proba(distance)
				distance_proba2 = self.extra_proba2(distance)
				distance_proba3 = self.extra_proba3(distance)
			else:
				distance = torch.cat([self.attribute_dict_embedding(x[:, 1]), self.attribute_dict_embedding(x[:, 2]),
									  torch.zeros((len(x), self.cell_feats.shape[-1])).float().to(device, non_blocking=True)], dim=-1)
				distance_proba = self.extra_proba(distance)
				distance_proba2 = self.extra_proba2(distance)
				distance_proba3 = self.extra_proba3(distance)
			
		else:
			distance_proba = torch.zeros((len(x), 1), dtype=torch.float, device=device)
			distance_proba2 = torch.zeros((len(x), 1), dtype=torch.float, device=device)
			distance_proba3 = torch.zeros((len(x), 1), dtype=torch.float, device=device)
		
		if not self.only_distance:
			# slf_attn_mask = get_attn_key_pad_mask(seq_k=x, seq_q=x)
			# non_pad_mask = get_non_pad_mask(x)
			
			dynamic, static = self.get_embedding(x, x_chrom, chroms_in_batch=chroms_in_batch)
			dynamic = self.layer_norm1(dynamic)
			static = self.layer_norm2(static)
			
			if self.diag_mask_flag:
				output = (dynamic - static) ** 2
			else:
				output = dynamic
			
			output_proba = self.pff_classifier_proba(static)
			# output_proba = torch.sum(output_proba * non_pad_mask, dim=-2, keepdim=False)
			# mask_sum = torch.sum(non_pad_mask, dim=-2, keepdim=False)
			# output_proba /= mask_sum
			output_proba = torch.mean(output_proba, dim=-2, keepdim=False)
			output_proba = output_proba + distance_proba
			
			
			output_mean = self.pff_classifier(output)
	
			# output_mean = torch.sum(output_mean * non_pad_mask, dim=-2, keepdim=False)
			# output_mean /= mask_sum
			output_mean = torch.mean(output_mean, dim=-2, keepdim=False)
			
			output_var = self.pff_classifier_var(static)
			# output_var = torch.sum(output_var * non_pad_mask, dim=-2, keepdim=False)
			# output_var /= mask_sum
			output_var = torch.mean(output_var, dim=-2, keepdim=False)
			output_mean = output_mean + distance_proba2
			output_var = output_var + distance_proba3
			
		else:
			return distance_proba2, distance_proba3, distance_proba
		return output_mean, output_var, output_proba


	def predict(self, input, input_chrom, verbose=False, batch_size=96, activation=None, extra_info=None):
		self.eval()
		with torch.no_grad():
			output = []
			if verbose:
				func1 = trange
			else:
				func1 = range
			if batch_size < 0:
				batch_size = len(input)
			with torch.no_grad():
				for j in func1(math.ceil(len(input) / batch_size)):
					x = input[j * batch_size:min((j + 1) * batch_size, len(input))]
					if type(input_chrom) is not tuple:
						x_chrom = input_chrom[j * batch_size:min((j + 1) * batch_size, len(input))]
						x_chrom = torch.from_numpy(x_chrom).long().to(device, non_blocking=True)
					else:
						a,b = input_chrom
						x_chrom = a[j * batch_size:min((j + 1) * batch_size, len(input))], b[j * batch_size:min((j + 1) * batch_size, len(input))]
						
					x = np2tensor_hyper(x, dtype=torch.long)
					
					
					if len(x.shape) == 1:
						x = pad_sequence(x, batch_first=True, padding_value=0).to(device, non_blocking=True)
					else:
						x = x.to(device, non_blocking=True)
					
					o, _, o_proba = self(x, x_chrom)

					if activation is not None:
						o = activation(o)

					if extra_info is not None:
						o = o * extra_info[x[:, 2] - x[:, 1]]

					output.append(o.detach().cpu())
			
			output = torch.cat(output, dim=0)
			torch.cuda.empty_cache()
		self.train()
		return output.numpy()


# A custom position-wise MLP.
# dims is a list, it would create multiple layer with tanh between them
# If dropout, it would add the dropout at the end. Before residual and
# layer-norm
class PositionwiseFeedForward(nn.Module):
	def __init__(
			self,
			dims,
			dropout=None,
			reshape=False,
			use_bias=True,
			residual=False,
			layer_norm=False):
		super(PositionwiseFeedForward, self).__init__()
		self.w_stack = []
		self.dims = dims
		for i in range(len(dims) - 1):
			self.w_stack.append(nn.Conv1d(dims[i], dims[i + 1], 1, bias=use_bias))
			# self.w_stack.append(nn.Linear(dims[i], dims[i + 1], bias=use_bias))
		
		self.w_stack = nn.ModuleList(self.w_stack)
		self.reshape = reshape
		self.layer_norm = nn.LayerNorm(dims[0])
		
		if dropout is not None:
			self.dropout = nn.Dropout(dropout)
		else:
			self.dropout = None
		
		self.residual = residual
		self.layer_norm_flag = layer_norm
		self.alpha = torch.nn.Parameter(torch.zeros(1))
		
		self.register_parameter("alpha", self.alpha)
	
	def forward(self, x):
		if self.layer_norm_flag:
			output = self.layer_norm(x)
		else:
			output = x
		output = output.transpose(1, 2)
		
		for i in range(len(self.w_stack) - 1):
			output = self.w_stack[i](output)
			output = activation_func(output)
			if self.dropout is not None:
				output = self.dropout(output)
		
		output = self.w_stack[-1](output)
		output = output.transpose(1, 2)
		
		if self.reshape:
			output = output.view(output.shape[0], -1, 1)
		
		if self.dims[0] == self.dims[-1]:
			# residual
			if self.residual:
				output = output + x
		
		return output


# A custom position wise MLP.
# dims is a list, it would create multiple layer with torch.tanh between them
# We don't do residual and layer-norm, because this is only used as the
# final classifier
class FeedForward(nn.Module):
	''' A two-feed-forward-layer module '''
	
	def __init__(self, dims, dropout=None, reshape=False, use_bias=True):
		super(FeedForward, self).__init__()
		self.w_stack = []
		for i in range(len(dims) - 1):
			self.w_stack.append(nn.Linear(dims[i], dims[i + 1], use_bias))
		
		self.w_stack = nn.ModuleList(self.w_stack)
		
		if dropout is not None:
			self.dropout = nn.Dropout(dropout)
		else:
			self.dropout = None
		
		self.reshape = reshape
	
	def forward(self, x):
		output = x
		for i in range(len(self.w_stack) - 1):
			output = self.w_stack[i](output)
			output = activation_func(output)
			if self.dropout is not None:
				output = self.dropout(output)
		output = self.w_stack[-1](output)
		
		if self.reshape:
			output = output.view(output.shape[0], -1, 1)
		
		return output


class ScaledDotProductAttention(nn.Module):
	''' Scaled Dot-Product Attention '''
	
	def __init__(self, temperature):
		super().__init__()
		self.temperature = temperature
	
	def masked_softmax(self, vector: torch.Tensor,
					   mask: torch.Tensor,
					   dim: int = -1,
					   memory_efficient: bool = False,
					   mask_fill_value: float = -1e32) -> torch.Tensor:
		
		if mask is None:
			result = torch.nn.functional.softmax(vector, dim=dim)
		else:
			mask = mask.float()
			while mask.dim() < vector.dim():
				mask = mask.unsqueeze(1)
			if not memory_efficient:
				# To limit numerical errors from large vector elements outside
				# the mask, we zero these out.
				result = torch.nn.functional.softmax(vector * mask, dim=dim)
				result = result * mask
				result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
			else:
				masked_vector = vector.masked_fill(
					(1 - mask).bool(), mask_fill_value)
				result = torch.nn.functional.softmax(masked_vector, dim=dim)
		return result
	
	def forward(self, q, k, v, diag_mask, mask=None):
		attn = torch.bmm(q, k.transpose(1, 2))
		attn = attn / self.temperature
		
		if mask is not None:
			attn = attn.masked_fill(mask, -float('inf'))
		
		attn = self.masked_softmax(
			attn, diag_mask, dim=-1, memory_efficient=False)
		
		output = torch.bmm(attn, v)
		
		return output, attn


class MultiHeadAttention(nn.Module):
	''' Multi-Head Attention module '''
	
	def __init__(
			self,
			n_head,
			d_model,
			d_k,
			d_v,
			dropout,
			diag_mask,
			input_dim):
		super().__init__()
		self.d_model = d_model
		self.input_dim = input_dim
		self.n_head = n_head
		self.d_k = d_k
		self.d_v = d_v
		
		self.w_qs = nn.Linear(input_dim, n_head * d_k, bias=False)
		self.w_ks = nn.Linear(input_dim, n_head * d_k, bias=False)
		self.w_vs = nn.Linear(input_dim, n_head * d_v, bias=False)
		
		nn.init.normal_(self.w_qs.weight, mean=0,
						std=np.sqrt(2.0 / (d_model + d_k)))
		nn.init.normal_(self.w_ks.weight, mean=0,
						std=np.sqrt(2.0 / (d_model + d_k)))
		nn.init.normal_(self.w_vs.weight, mean=0,
						std=np.sqrt(2.0 / (d_model + d_v)))
		
		self.attention = ScaledDotProductAttention(
			temperature=np.power(d_k, 0.5))
		
		self.fc1 = FeedForward([n_head * d_v, d_model], use_bias=False)
		self.fc2 = FeedForward([n_head * d_v, d_model], use_bias=False)
		
		self.layer_norm1 = nn.LayerNorm(input_dim)
		self.layer_norm2 = nn.LayerNorm(input_dim)
		self.layer_norm3 = nn.LayerNorm(input_dim)
		
		if dropout is not None:
			self.dropout = nn.Dropout(dropout)
		else:
			self.dropout = dropout
		
		self.diag_mask_flag = diag_mask
		self.diag_mask = None
		
		self.alpha_static = torch.nn.Parameter(torch.zeros(1))
		self.alpha_dynamic = torch.nn.Parameter(torch.zeros(1))

		self.register_parameter("alpha_static", self.alpha_static)
		self.register_parameter("alpha_dynamic", self.alpha_dynamic)

	def forward(self, q, k, v, diag_mask=None, mask=None):
		d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
		
		residual_dynamic = q
		residual_static = v
		
		q = self.layer_norm1(q)
		k = self.layer_norm2(k)
		v = self.layer_norm3(v)
		
		sz_b, len_q, _ = q.shape
		sz_b, len_k, _ = k.shape
		sz_b, len_v, _ = v.shape
		
		q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
		k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
		v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
		
		q = q.permute(2, 0, 1, 3).contiguous(
		).view(-1, len_q, d_k)  # (n*b) x lq x dk
		k = k.permute(2, 0, 1, 3).contiguous(
		).view(-1, len_k, d_k)  # (n*b) x lk x dk
		v = v.permute(2, 0, 1, 3).contiguous(
		).view(-1, len_v, d_v)  # (n*b) x lv x dv
		
		n = sz_b * n_head
		
		if self.diag_mask is not None:
			if (len(self.diag_mask) <= n) or (
					self.diag_mask.shape[1] != len_v):
				self.diag_mask = torch.ones((len_v, len_v), device=device)
				if self.diag_mask_flag:
					self.diag_mask -= torch.eye(len_v, len_v, device=device)
				self.diag_mask = self.diag_mask.repeat(n, 1, 1).bool()
				diag_mask = self.diag_mask
			else:
				diag_mask = self.diag_mask[:n]
		
		else:
			self.diag_mask = (torch.ones((len_v, len_v), device=device))
			if self.diag_mask_flag:
				self.diag_mask -= torch.eye(len_v, len_v, device=device)
			self.diag_mask = self.diag_mask.repeat(n, 1, 1).bool()
			diag_mask = self.diag_mask
		
		if mask is not None:
			mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
		
		dynamic, attn = self.attention(q, k, v, diag_mask, mask=mask)
		
		dynamic = dynamic.view(n_head, sz_b, len_q, d_v)
		dynamic = dynamic.permute(
			1, 2, 0, 3).contiguous().view(
			sz_b, len_q, -1)  # b x lq x (n*dv)
		static = v.view(n_head, sz_b, len_q, d_v)
		static = static.permute(
			1, 2, 0, 3).contiguous().view(
			sz_b, len_q, -1)  # b x lq x (n*dv)
		
		dynamic = self.dropout(self.fc1(dynamic)) if self.dropout is not None else self.fc1(dynamic)
		static = self.dropout(self.fc2(static)) if self.dropout is not None else self.fc2(static)
		
		dynamic = dynamic  # + residual_dynamic
		
		static = static  # + residual_static
		
		return dynamic, static, attn


class EncoderLayer(nn.Module):
	'''A self-attention layer + 2 layered pff'''
	
	def __init__(
			self,
			n_head,
			d_model,
			d_k,
			d_v,
			dropout_mul,
			dropout_pff,
			diag_mask,
			bottle_neck,
			dynamic_nn=None,
			static_nn=None):
		super().__init__()
		self.n_head = n_head
		self.d_k = d_k
		self.d_v = d_v
		
		self.mul_head_attn = MultiHeadAttention(
			n_head,
			d_model,
			d_k,
			d_v,
			dropout=dropout_mul,
			diag_mask=diag_mask,
			input_dim=bottle_neck)
		self.pff_n1 = PositionwiseFeedForward(
			[d_model, d_model, d_model], dropout=dropout_pff, residual=True, layer_norm=True)
		
		residual = True if bottle_neck == d_model else False
		self.pff_n2 = PositionwiseFeedForward(
			[bottle_neck, d_model, d_model], dropout=dropout_pff, residual=residual, layer_norm=True)
		self.dynamic_nn = dynamic_nn
		self.static_nn = static_nn
		self.dropout = nn.Dropout(0.2)
		
	def forward(self, dynamic, static, chrom_info, slf_attn_mask, non_pad_mask, chroms_in_batch):
		
		if type(chrom_info) is tuple:
			chrom_info, to_neighs = chrom_info
		else:
			to_neighs = chrom_info
			
		final_chroms_in_batch = None
		if chroms_in_batch is not None:
			if len(chroms_in_batch) == 1:
				# only one chromosome, use route_nn
				final_chroms_in_batch = int(chroms_in_batch[0])
		
		if isinstance(self.dynamic_nn, GraphSageEncoder_with_weights) :
			dynamic, static = self.dynamic_nn(dynamic, to_neighs, route_nn=final_chroms_in_batch)
		else:
			static_cell = self.static_nn(static[:, 0], to_neighs, route_nn=0)
			static_bin = self.static_nn(static[:, 1:].contiguous(), to_neighs, route_nn=final_chroms_in_batch)
			static = torch.cat([static_cell[:, None, :], static_bin], dim=1)
			dynamic = static
			
		dynamic, static, attn = self.mul_head_attn(
			dynamic, dynamic, static)
		dynamic = self.pff_n1(dynamic) #* non_pad_mask
		# static = self.pff_n2(static * non_pad_mask) * non_pad_mask
		return dynamic, static

# Sampling positive triplets.
# The number of triplets from each chromosome is balanced across different chromosome
class DataGenerator():
	def __init__(self, edges, edge_chrom, edge_weight, batch_size, flag=False, num_list=None, k=1):
		self.batch_size = batch_size
		self.flag = flag
		self.k = k
		self.batch_size = int(self.batch_size)
		self.num_list = list(num_list)
		
		self.edges = edges
		self.edge_weight = edge_weight
		self.edge_chrom = edge_chrom
		self.chrom_list = np.arange(len(self.num_list) - 1)
		self.size_list = []
		
		print ("initializing data generator")
		for i in trange(len(self.num_list) - 1):
			self.size_list.append(len(self.edges[i]))
			if len(self.edges[i]) == 0:
				print ("The %d th chrom in your chrom_list has no sample in this generator" % i)
				continue
				
			while len(self.edges[i]) <= (self.batch_size):
				self.edges[i] = np.concatenate([self.edges[i], self.edges[i]])
				self.edge_weight[i] = np.concatenate([self.edge_weight[i], self.edge_weight[i]])
				self.edge_chrom[i] = np.concatenate([self.edge_chrom[i], self.edge_chrom[i]])
		
		self.pointer = np.zeros(int(np.max(self.chrom_list) + 1)).astype('int')
		self.size_list /= np.sum(self.size_list)
		
	def filter_edges(self, min_bin=0, max_bin=-1):
		for i in trange(len(self.edges)):
			mask = ((self.edges[i][:, 2] - self.edges[i][:, 1]) > min_bin) & (
						(self.edges[i][:, 2] - self.edges[i][:, 1]) < max_bin)
			self.edges[i] = self.edges[i][mask]
			self.edge_weight[i] = self.edge_weight[i][mask]
			self.edge_chrom[i] = self.edge_chrom[i][mask]
			
		
		
	def next_iter(self):
		chroms = np.random.choice(self.chrom_list, size=self.k, replace=True)
		e_list = []
		c_list = []
		w_list = []
		
		batch_size = self.batch_size / self.k
		batch_size = int(batch_size)
		
		for chrom in chroms:
			if len(self.edges[chrom]) == 0:
				continue
				
			self.pointer[chrom] += batch_size
			
			if self.pointer[chrom] > len(self.edges[chrom]):
				index = np.random.permutation(len(self.edges[chrom]))
				self.edges[chrom] = (self.edges[chrom])[index]
				self.edge_weight[chrom] = (self.edge_weight[chrom])[index]
				self.edge_chrom[chrom] = (self.edge_chrom[chrom])[index]
				self.pointer[chrom] = batch_size
			
			index = range(self.pointer[chrom] - batch_size, min(self.pointer[chrom], len(self.edges[chrom])))
			e, c, w = (self.edges[chrom])[index], (self.edge_chrom[chrom])[index], (self.edge_weight[chrom])[index]
			e_list.append(e)
			c_list.append(c)
			w_list.append(w)
		e = np.concatenate(e_list, axis=0)
		c = np.concatenate(c_list, axis=0)
		w = np.concatenate(w_list, axis=0)
		return e, c, w, chroms


class MeanAggregator(nn.Module):
	"""
	Aggregates a node's embeddings using mean of neighbors' embeddings
	"""
	
	def __init__(self, features, gcn=False, num_list=None, start_end_dict=None, pass_pseudo_id=False):
		"""
		Initializes the aggregator for a specific graph.
		features -- function mapping LongTensor of node ids to FloatTensor of feature values.
		gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
		"""
		
		super(MeanAggregator, self).__init__()
		
		self.features = features
		self.gcn = gcn
		self.num_list = torch.as_tensor(num_list)
		self.mask = None
		self.start_end_dict = start_end_dict
		
		# If the feature function comes from a graphsage encoder, use the cell_id * (bin_num+1) + bin_id as the bin_id
		self.pass_pseudo_id = pass_pseudo_id
		
		print("pass_pseudo_id", self.pass_pseudo_id)
	
	# nodes_real represents the true bin_id, nodes might represent the pseudo_id generated by cell_id * (bin_num+1) + bin_id
	def forward(self, nodes_real, to_neighs, num_sample=10):
		"""
		nodes --- list of nodes in a batch
		to_neighs --- list of sets, each set is the set of neighbors for node in batch
		num_sample --- number of neighbors to sample. No sampling if None.
		"""
		
		samp_neighs = np.array(to_neighs)
		
		unique_nodes = {}
		unique_nodes_list = []
		
		count = 0
		column_indices = []
		row_indices = []
		v = []
		
		for i, samp_neigh in enumerate(samp_neighs):
			samp_neigh = set(samp_neigh)
			for n in samp_neigh:
				if n not in unique_nodes:
					unique_nodes[n] = count
					unique_nodes_list.append(n)
					count += 1
				
				column_indices.append(unique_nodes[n])
				row_indices.append(i)
				v.append(1 / len(samp_neigh))
		
		unique_nodes_list = torch.LongTensor(unique_nodes_list).to(device, non_blocking=True)
		
		mask = torch.sparse.FloatTensor(torch.LongTensor([row_indices, column_indices]),
										torch.tensor(v, dtype=torch.float),
										torch.Size([len(samp_neighs), len(unique_nodes_list)])).to(device, non_blocking=True)
		
		
		embed_matrix = self.features(unique_nodes_list)
		
		to_feats = mask.mm(embed_matrix)
		return to_feats
	

class MeanAggregator_with_weights(nn.Module):
	"""
	Aggregates a node's embeddings using mean of neighbors' embeddings
	"""
	
	def __init__(self, features, gcn=False, num_list=None, start_end_dict=None, pass_pseudo_id=False, remove=False, pass_remove=False):
		"""
		Initializes the aggregator for a specific graph.
		features -- function mapping LongTensor of node ids to FloatTensor of feature values.
		gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
		"""
		
		super(MeanAggregator_with_weights, self).__init__()
		
		self.features = features
		self.gcn = gcn
		self.num_list = torch.as_tensor(num_list)
		self.mask = None
		self.start_end_dict = start_end_dict
		
		# If the feature function comes from a graphsage encoder, use the cell_id * (bin_num+1) + bin_id as the bin_id

		self.pass_pseudo_id = pass_pseudo_id
		self.remove=remove
		self.pass_remove = pass_remove
		print("pass_pseudo_id", self.pass_pseudo_id)
	
	@staticmethod
	def list_pass(x, num_samples):
		return x
	
	# nodes_real represents the true bin_id, nodes might represent the pseudo_id generated by cell_id * (bin_num+1) + bin_id
	def forward(self, nodes_real, to_neighs, num_sample=10, route_nn=None):
		"""
		nodes --- list of nodes in a batch
		to_neighs --- list of sets, each set is the set of neighbors for node in batch
		num_sample --- number of neighbors to sample. No sampling if None.
		"""
		
		
		# if hasattr(torch, "sparse_csr_tensor"):
		# 	crow_indices, column_indices, v, unique_nodes_list = to_neighs
		# 	unique_nodes_list = unique_nodes_list.to(device, non_blocking=True)
		# 	mask = torch.sparse_csr_tensor(torch.as_tensor(crow_indices).long().to(device),
		# 	                               torch.as_tensor(column_indices).long().to(device),
		# 	                               torch.from_numpy(v).float().to(device),
		# 	                               size=torch.Size([len(nodes_real),
		# 	                                           len(unique_nodes_list)]), device=device, requires_grad=False)
		# else:
		indices,  v, unique_nodes_list = to_neighs
		unique_nodes_list = unique_nodes_list.to(device, non_blocking=True)
		mask = torch.sparse_coo_tensor(indices.to(device), v.to(device),
		                               torch.Size([len(nodes_real),
		                                           len(unique_nodes_list)]), device=device)
		embed_matrix = self.features(unique_nodes_list, route_nn=route_nn)
		to_feats = mask.mm(embed_matrix)

		return to_feats

	def forward_GCN(self, nodes, adj, moving_range=0, route_nn=None):
		embed_matrix = self.features(nodes, route_nn=route_nn)
		indices, data, shape = adj
		mask = torch.sparse_coo_tensor(indices.to(device), data.to(device),
		                               torch.Size([shape[0], shape[1]]), device=device)
		to_feats = mask.mm(embed_matrix)
		
		return to_feats


def moving_avg(adj, moving_range):
	adj_origin = adj.copy()
	adj = adj.copy()
	adj = adj * norm.pdf(0)
	for i in range(moving_range * 2):
		before_list = [adj_origin[0, :]] * (i+1) + [adj_origin[:-(i+1), :]]
		adj_before = vstack(before_list)
		after_list = [adj_origin[i+1:, :]] + [adj_origin[-1, :]] * (i+1)
		adj_after = vstack(after_list)
		adj = adj + (adj_after + adj_before) * norm.pdf((i+1) / moving_range)
	return adj


class GraphSageEncoder_with_weights(nn.Module):
	"""
	Encodes a node's using 'convolutional' GraphSage approach
	"""
	
	def __init__(self, features, linear_features=None, feature_dim=64,
				 embed_dim=64,
				 num_sample=10, gcn=False, num_list=None, transfer_range=0, start_end_dict=None, pass_pseudo_id=False,
				 remove=False, pass_remove=False):
		super(GraphSageEncoder_with_weights, self).__init__()
		
		self.features = features
		self.linear_features = linear_features
		self.feat_dim = feature_dim
		self.pass_pseudo_id = pass_pseudo_id
		
		# aggregator aggregates through hic graph
		self.aggregator = MeanAggregator_with_weights(self.features, gcn, num_list, start_end_dict, pass_pseudo_id, remove, pass_remove)
		# linear aggregator aggregats through 1D genomic neighbors
		self.linear_aggregator = MeanAggregator(self.linear_features, gcn, num_list, start_end_dict, pass_pseudo_id)
		
		self.num_sample = num_sample
		self.transfer_range = transfer_range
		self.gcn = gcn
		self.embed_dim = embed_dim
		self.start_end_dict = start_end_dict
		
		input_size = 1
		if not self.gcn:
			input_size += 1
		if self.transfer_range > 0:
			input_size += 1
		
		self.nn = nn.Linear(input_size * self.feat_dim, embed_dim)
		self.num_list = torch.as_tensor(num_list)
		self.bin_feats = torch.zeros([int(self.num_list[-1]) + 1, self.feat_dim*input_size], dtype=torch.float, device=device)
		self.fix = False
		self.forward = self.forward_on_hook
		
	def start_fix(self):
		self.fix = True
		ids = (torch.arange(int(self.num_list[0])) + 1).long().to(device, non_blocking=True).view(-1)
		self.cell_feats = self.features(ids)
	
	
	def fix_cell2(self, cell, bin_ids=None, sparse_matrix=None, local_transfer_range=0, route_nn_list=None):
		self.fix = True
		self.eval()
		input_size = 1
		if not self.gcn:
			input_size += 1
		if self.transfer_range > 0:
			input_size += 1
		self.bin_feats = torch.zeros([int(self.num_list[-1]) + 1, self.feat_dim * input_size], dtype=torch.float,
		                             device=device)
		# instead of
		with torch.no_grad():
			for chrom, bin_id in enumerate(bin_ids):
				nodes_flatten = torch.from_numpy(bin_id).long().to(device, non_blocking=True)
				neigh_feats = self.aggregator.forward_GCN(nodes_flatten,
													  sparse_matrix[chrom], local_transfer_range, route_nn=route_nn_list[chrom])
				tr = self.transfer_range
				if tr > 0:
					start = np.maximum(bin_id - tr, self.start_end_dict[bin_id, 0] + 1)
					end = np.minimum(bin_id + tr, self.start_end_dict[bin_id, 1] + 1)
					
					to_neighs = np.array([list(range(s, e)) for s, e in zip(start, end)], dtype='object')
					
					neigh_feats_linear = self.linear_aggregator.forward(nodes_flatten,
																		to_neighs,
																		2 * tr + 1)
				list1 = [neigh_feats, neigh_feats_linear] if tr > 0 else [neigh_feats]
				if not self.gcn:
					bin_feats_self = self.features(nodes_flatten)
					list1.append(bin_feats_self)
				
					
				if len(list1) > 0:
					combined = torch.cat(list1, dim=-1)
				else:
					combined = list1[0]
				
				self.bin_feats[nodes_flatten] = combined.detach().clone()
	
	
	def forward_off_hook(self, nodes, to_neighs, *args, route_nn=None):
		if len(nodes.shape) == 1:
			nodes_flatten = nodes
		else:
			sz_b, len_seq = nodes.shape
			nodes_flatten = nodes[:, 1:].contiguous().view(-1)
			
		cell_feats = self.cell_feats[nodes[:, 0] - 1, :]
		neigh_feats = activation_func(self.nn(self.bin_feats[nodes_flatten, :])).view(sz_b, len_seq - 1, -1)
		return torch.cat([cell_feats[:, None, :], neigh_feats], dim=1).view(sz_b, len_seq, -1), \
		       torch.cat([cell_feats[:, None, :], self.features(nodes_flatten).view(sz_b, len_seq - 1, -1)
					], dim=1).view(sz_b, len_seq, -1)
	
	def forward_on_hook(self, nodes, to_neighs, *args, route_nn=None):
		"""
		Generates embeddings for a batch of nodes.
		nodes     -- list of nodes
		pseudo_nodes -- pseudo_nodes for getting the correct neighbors
		"""
		
		tr = self.transfer_range
		
		if len(nodes.shape) == 1:
			nodes_flatten = nodes
		else:
			sz_b, len_seq = nodes.shape
			nodes_flatten = nodes[:, 1:].contiguous().view(-1)
		
		if self.fix:
			cell_feats = self.cell_feats[nodes[:, 0] - 1, :]
			neigh_feats = self.bin_feats[nodes_flatten, :].view(sz_b, len_seq - 1, -1)
			if tr > 0:
				neigh_feats_linear = self.bin_feats_linear[nodes_flatten, :].view(sz_b, len_seq - 1, -1)
		
		
		else:
			if len(nodes.shape) == 1:

				
				neigh_feats = self.aggregator.forward(nodes_flatten, to_neighs, self.num_sample, route_nn=route_nn)
			else:
				cell_feats = self.features(nodes[:, 0].to(device, non_blocking=True), route_nn=0)
				neigh_feats = self.aggregator.forward(nodes_flatten, to_neighs,
													  self.num_sample).view(sz_b, len_seq - 1, -1)
			if tr > 0:
				nodes_flatten_np = nodes_flatten.cpu().numpy()
				start = np.maximum(nodes_flatten_np - tr, self.start_end_dict[nodes_flatten_np, 0])
				end = np.minimum(nodes_flatten_np + tr, self.start_end_dict[nodes_flatten_np, 1])
				
				to_neighs = np.array([list(range(s, e)) for s, e in zip(start, end)])
				neigh_feats_linear = self.linear_aggregator.forward(nodes_flatten,
																	to_neighs,
																	2 * tr + 1)
				if len(nodes.shape) > 1:
					neigh_feats_linear = neigh_feats_linear.view(sz_b, len_seq - 1, -1)
				
		
		list1 = [neigh_feats, neigh_feats_linear] if tr > 0 else [neigh_feats]
		
		if not self.gcn:
			if self.fix:
				self_feats = self.bin_feats_self[nodes_flatten].view(sz_b, len_seq - 1, -1)
			else:
				if len(nodes.shape) == 1:
					self_feats = self.features(nodes_flatten, route_nn=route_nn)
				else:
					sz_b, len_seq = nodes.shape
					self_feats = self.features(nodes_flatten, route_nn=route_nn).view(sz_b, len_seq - 1, -1)
					
			list1.append(self_feats)
		
		if len(list1) > 0:
			combined = torch.cat(list1, dim=-1)
		else:
			combined = list1[0]
		
		combined = activation_func(self.nn(combined))
		
		if len(nodes.shape) > 1:
			combined = torch.cat([cell_feats[:, None, :], combined], dim=1).view(sz_b, len_seq, -1)
		
		return combined, torch.cat([cell_feats[:, None, :], self_feats], dim=1).view(sz_b, len_seq, -1)


