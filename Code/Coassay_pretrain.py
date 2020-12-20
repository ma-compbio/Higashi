
import sys

import os

from Higashi_backend.Modules import *
from Higashi_backend.utils import *
from sklearn.preprocessing import StandardScaler

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


def log_cosh(pred, truth, sample_weight=None):
	ey_t = truth - pred
	if sample_weight is not None:
		
		return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)) * sample_weight)
	else:
		return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


def forward_batch(model, batch_size=128):
	batch = torch.randperm(len(train_cell_ids))[:batch_size]
	adj = hic[train_cell_ids[batch]]
	targets = cell_attributes[train_cell_ids[batch]]
	embed, recon = model(adj, return_recon=True)
	mse_loss = log_cosh(recon, targets)
	return mse_loss, embed


if __name__ == '__main__':
	
	config, chrom = sys.argv[1], sys.argv[2]
	config = get_config(config)
	
	temp_dir = config['temp_dir']
	
	get_free_gpu()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	hic = np.array(np.load(os.path.join(temp_dir, "%s_cell_feats.npy" % chrom), allow_pickle=True))
	dim = int(math.sqrt(hic.shape[-1]) * 2)
	hic = StandardScaler().fit_transform(hic)
	hic = torch.from_numpy(hic).to(device).float()
	coassay = np.load(os.path.join(temp_dir, "coassay_%s.npy" % chrom), allow_pickle=True)
	
	print("chrom", chrom, hic.shape, coassay.shape)
	cell_attributes = torch.from_numpy(coassay).to(device).float()
	cell_ids = torch.arange(len(hic)).long().to(device)
	train_cell_ids = torch.randperm(len(cell_ids)).long().to(device)
	
	model = AutoEncoder([hic.shape[-1], dim * 4, dim * 2, dim], [dim, cell_attributes.shape[-1]],
	                    add_activation=True)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	
	for i in range(50000):
		loss, embed = forward_batch(model, len(hic))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		if i % 1000 == 0:
			print("Pre-training co-assay", chrom, i, loss.item())
			
	print(chrom, "finish", loss.item())
	adj = hic[cell_ids]
	embed = model(adj, return_recon=False)
	np.save(os.path.join(temp_dir, "pretrain_coassay_%s.npy" % chrom), embed.detach().cpu().numpy())