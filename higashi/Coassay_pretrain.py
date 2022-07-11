
import sys

import os

from Higashi_backend.Modules import *
from Higashi_backend.utils import *
from sklearn.preprocessing import StandardScaler
torch.backends.cudnn.benchmark = True
from sklearn.decomposition import PCA
from torch.optim.lr_scheduler import ReduceLROnPlateau
# One process for pretrain of the coassay 1D signal task

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


def XTanhLoss( y_t, y_prime_t):
	ey_t = y_t - y_prime_t
	return torch.mean(ey_t * torch.tanh(ey_t))



def XSigmoidLoss(y_t, y_prime_t):
	ey_t = y_t - y_prime_t
	# return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)
	return torch.mean(2 * ey_t * torch.sigmoid(ey_t) - ey_t)
	
	
def log_cosh(pred, truth, sample_weight=None):
	# ey_t = truth - pred
	# loss = torch.log(torch.clamp(torch.cosh(ey_t), 1.0, 100))
	# if torch.sum(torch.isnan(loss))>0:
	# 	print (chrom, pred[torch.isnan(loss)])
	# return torch.mean(loss)
	ey_t = truth - pred
	return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


def forward_batch(model, batch_size=128):
	batch = torch.randperm(len(train_cell_ids))[:batch_size]
	adj = hic[train_cell_ids[batch]]
	targets = cell_attributes[train_cell_ids[batch]]
	embed, recon = model(adj, return_recon=True)
	mse_loss = XSigmoidLoss(recon, targets)
	return mse_loss, embed


if __name__ == '__main__':
	
	config, chrom = sys.argv[1], sys.argv[2]
	config = get_config(config)
	
	temp_dir = config['temp_dir']
	chrom_list = config['chrom_list']
	
	get_free_gpu()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	with h5py.File(os.path.join(temp_dir, "node_feats.hdf5"), "r") as save_file:
		hic = np.array(save_file["cell"]["%d" % chrom_list.index(chrom)])
		
	square_size = int(math.sqrt(hic.shape[-1]))
	dim = square_size * 2
	# hic = StandardScaler().fit_transform(hic)
	
	hic[np.isnan(hic)] = 0.0
	if hic.shape[-1] > hic.shape[0] * 3:
		hic = PCA(n_components = int(np.min(hic.shape)-1)).fit_transform(hic)
	hic = StandardScaler().fit_transform(hic)
	# hic = hic.reshape((len(hic), square_size, square_size))
	
	hic = torch.from_numpy(hic).to(device).float()

	coassay = np.load(os.path.join(temp_dir, "temp", "coassay_%s.npy" % chrom), allow_pickle=True)
	
	print("chrom", chrom, hic.shape, coassay.shape)
	cell_attributes = torch.from_numpy(coassay).to(device).float()
	cell_ids = torch.arange(len(hic)).long().to(device)
	train_cell_ids = torch.randperm(len(cell_ids)).long().to(device)
	
	model = AutoEncoder([hic.shape[-1], dim * 4,  dim * 2, dim], [dim, cell_attributes.shape[-1]],
						add_activation=True, layer_norm=True).to(device)
	
	for name, param in model.named_parameters():
		print(name, param.requires_grad, param.shape)
	
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	# scheduler = ReduceLROnPlateau(optimizer, 'min')
	loss_prev = None
	no_improv_count = 0
	bar = trange(50000, desc=' - (Training) ', leave=False, )
	for i in range(50000):
		loss, embed = forward_batch(model, len(hic))
		if loss_prev is None:
			loss_prev = loss.item()
		else:
			if loss.item() <= loss_prev-0.001:
				loss_prev = loss.item()
				no_improv_count = 0
			else:
				no_improv_count += 1
		# if (no_improv_count >= 30000):
		# 	break
		optimizer.zero_grad()
		loss.backward()
		# torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		
		optimizer.step()
		# scheduler.step(loss)
		
		bar.update(n=1)
		# if i % 1000 == 0:
		bar.set_description("Pre-training co-assay %s, Loss:%.4f, best Loss:%.4f" %
							(chrom,  loss.item(), loss_prev),
							refresh=True)
			
	torch.cuda.empty_cache()
	print(chrom, "finish", loss.item())
	adj = hic[cell_ids]
	embed, recon_ = model(adj, return_recon=True)
	np.save(os.path.join(temp_dir, "temp", "pretrain_coassay_%s.npy" % chrom), embed.detach().cpu().numpy())
	np.save(os.path.join(temp_dir, "temp", "pretrain_coassay_recon_%s.npy" % chrom), recon_.detach().cpu().numpy())