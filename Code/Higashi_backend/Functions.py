import torch.nn as nn
import torch.nn.functional as F
import torch

def XSigmoidLoss(y_t, y_prime_t):
	ey_t = y_t - y_prime_t
	# return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)
	return torch.mean(2 * ey_t * torch.sigmoid(ey_t) - ey_t)


def arcosh(x):
	return torch.log(x + torch.sqrt(x ** 2 + 1))

def swish(x):
	return x * torch.sigmoid(x)


def sparse_autoencoder_error(y_true, y_pred, sparse_rate):
	return torch.mean(torch.sum(((torch.sign(y_true) * (y_true - y_pred)) ** 2) * sparse_rate, dim=-1) +
					  torch.sum(((y_true == 0).float() * (y_true - y_pred)) ** 2, dim=-1))


def get_non_pad_mask(seq):
	assert seq.dim() == 2
	return seq.ne(0).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
	''' For masking out the padding part of key sequence. '''
	
	# Expand to fit the shape of key query attention matrix.
	len_q = seq_q.size(1)
	padding_mask = seq_k.eq(0)
	padding_mask = padding_mask.unsqueeze(
		1).expand(-1, len_q, -1)  # b x lq x lk
	
	return padding_mask


def spy_sparse2torch_sparse(data):
	"""
	:param data: a scipy sparse csr matrix
	:return: a sparse torch tensor
	"""
	samples=data.shape[0]
	features=data.shape[1]
	values=data.data
	coo_data=data.tocoo()
	indices=torch.LongTensor([coo_data.row,coo_data.col])
	t=torch.sparse.FloatTensor(indices,torch.from_numpy(values).float(),[samples,features])
	return t