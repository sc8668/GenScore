import torch as th
import numpy as np
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import sys
sys.path.append("/home/shenchao/rtmscorepyg/code")
from GenScore.data.data import PDBbindDataset
from GenScore.model.model import GenScore, GraphTransformer, GatedGCN
from GenScore.model.utils import EarlyStopping, set_random_seed, run_a_train_epoch, run_an_eval_epoch, mdn_loss_fn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')



if __name__ == '__main__':
	import argparse
	p = argparse.ArgumentParser()
	p.add_argument('--num_epochs', type=int, default=5000)
	p.add_argument('--batch_size', type=int, default=64) 
	p.add_argument('--aux_weight', type=float, default=0.001)
	p.add_argument('--affi_weight', type=float, default=-0.5)
	p.add_argument('--patience', type=int, default=70)
	p.add_argument('--num_workers', type=int, default=8)
	p.add_argument('--model_path', type=str, default="xxx.pth")
	p.add_argument('--encoder', type=str, choices=['gt', 'gatedgcn'], default="gt")
	p.add_argument('--mode', type=str, choices=['lower', 'higher'], default="lower")
	p.add_argument('--finetune', action="store_true", default=False)
	p.add_argument('--original_model_path', type=str, default=None)
	p.add_argument('--lr', type=int, default=3)
	p.add_argument('--weight_decay', type=int, default=5)
	p.add_argument('--data_dir', type=str, default="/home/shenchao/rtmscorepyg/code/dataset")
	p.add_argument('--data_prefix', type=str, default="v2020_train")
	p.add_argument('--valnum', type=int, default=1500)
	p.add_argument('--seeds', type=int, default=126)
	p.add_argument('--hidden_dim0', type=int, default=128)
	p.add_argument('--hidden_dim', type=int, default=128)
	p.add_argument('--n_gaussians', type=int, default=10)
	p.add_argument('--dropout_rate', type=float, default=0.15)
	p.add_argument('--dist_threhold', type=float, default=7., help="the distance threhold for training")
	p.add_argument('--dist_threhold2', type=float, default=5., help="the distance threhold for testing")
	#p.add_argument('--device', type=str, default="cpu")
	args = p.parse_args()
	args.device = 'cuda' if th.cuda.is_available() else 'cpu'
	
	data = PDBbindDataset(ids="%s/%s_ids.npy"%(args.data_dir, args.data_prefix),
					  ligs="%s/%s_lig.pt"%(args.data_dir, args.data_prefix),
					  prots="%s/%s_prot.pt"%(args.data_dir, args.data_prefix)
					  )
	
	train_inds, val_inds = data.train_and_test_split(valnum=args.valnum, seed=args.seeds)
	train_data = PDBbindDataset(ids=data.pdbids[train_inds],
								ligs=data.gls[train_inds],
								prots=data.gps[train_inds],
								labels=data.labels[train_inds]
								)
	val_data = PDBbindDataset(ids=data.pdbids[val_inds],
								ligs=data.gls[val_inds],
								prots=data.gps[val_inds],
								labels=data.labels[val_inds]
								)
	if args.encoder == "gt":		
		ligmodel = GraphTransformer(in_channels=41, 
									edge_features=10, 
									num_hidden_channels=args.hidden_dim0, 
									activ_fn=th.nn.SiLU(),
									transformer_residual=True,
									num_attention_heads=4,
									norm_to_apply='batch',
									dropout_rate=0.15,
									num_layers=6
									)
		
		protmodel = GraphTransformer(in_channels=41, 
									edge_features=5, 
									num_hidden_channels=args.hidden_dim0, 
									activ_fn=th.nn.SiLU(),
									transformer_residual=True,
									num_attention_heads=4,
									norm_to_apply='batch',
									dropout_rate=0.15,
									num_layers=6
									)
	else:		
		ligmodel = GatedGCN(in_channels=41, 
							edge_features=10, 
							num_hidden_channels=args.hidden_dim0, 
							residual=True,
							dropout_rate=0.15,
							equivstable_pe=False,
							num_layers=6
							)
		
		protmodel = GatedGCN(in_channels=41, 
							edge_features=5, 
							num_hidden_channels=args.hidden_dim0, 
							residual=True,
							dropout_rate=0.15,
							equivstable_pe=False,
							num_layers=6
							)
	
	
	model = GenScore(ligmodel, protmodel, 
					in_channels=args.hidden_dim0, 
					hidden_dim=args.hidden_dim, 
					n_gaussians=args.n_gaussians, 
					dropout_rate=args.dropout_rate,
					dist_threhold=args.dist_threhold).to(args.device)
	
	if args.finetune:
		if args.original_model_path is None:
			raise ValueError('the argument "original_model_path" should be given')
		checkpoint = th.load(args.original_model_path, map_location=th.device(args.device))
		model.load_state_dict(checkpoint['model_state_dict']) 
	
	optimizer = th.optim.Adam(model.parameters(), lr=10**-args.lr, weight_decay=10**-args.weight_decay)
	
	train_loader = DataLoader(dataset=train_data, 
							batch_size=args.batch_size,
							shuffle=True, 
							num_workers=args.num_workers)
	
	val_loader = DataLoader(dataset=val_data, 
							batch_size=args.batch_size,
							shuffle=False, 
							num_workers=args.num_workers)
	
	stopper = EarlyStopping(patience=args.patience, mode=args.mode, filename=args.model_path)

	set_random_seed(args.seeds)
	for epoch in range(args.num_epochs):	
		# Train
		total_loss_train, mdn_loss_train, affi_loss_train, atom_loss_train, bond_loss_train = run_a_train_epoch(epoch, model, train_loader, optimizer, affi_weight=args.affi_weight, aux_weight=args.aux_weight, dist_threhold=args.dist_threhold2, device=args.device)	
		if np.isinf(mdn_loss_train) or np.isnan(mdn_loss_train): 
			print('Inf ERROR')
			break
		# Validation and early stop
		total_loss_val, mdn_loss_val, affi_loss_val, atom_loss_val, bond_loss_val = run_an_eval_epoch(model, val_loader, dist_threhold=args.dist_threhold2, affi_weight=args.affi_weight, aux_weight=args.aux_weight, device=args.device)
		early_stop = stopper.step(total_loss_val, model)
		print('epoch {:d}/{:d}, total_loss_val {:.4f}, mdn_loss_val {:.4f}, affi_loss_val {:.4f}, atom_loss_val {:.4f}, bond_loss_val {:.4f}, best validation {:.4f}'.format(epoch + 1, args.num_epochs, total_loss_val, mdn_loss_val, affi_loss_val, atom_loss_val, bond_loss_val, stopper.best_score)) #+' validation result:', validation_result)
		if early_stop:
			break
	
	
	stopper.load_checkpoint(model)
	total_loss_train, mdn_loss_train, affi_loss_train, atom_loss_train, bond_loss_train = run_an_eval_epoch(model, train_loader, dist_threhold=args.dist_threhold2, affi_weight=args.affi_weight, aux_weight=args.aux_weight, device=args.device)
	total_loss_val, mdn_loss_val, affi_loss_val, atom_loss_val, bond_loss_val = run_an_eval_epoch(model, val_loader, dist_threhold=args.dist_threhold2, affi_weight=args.affi_weight, aux_weight=args.aux_weight, device=args.device)		
	
	print("total_loss_train:%s, mdn_loss_train:%s, affi_loss_train:%s, atom_loss_train:%s, bond_loss_train:%s"%(total_loss_train, mdn_loss_train, affi_loss_train, atom_loss_train, bond_loss_train))
	print("total_loss_val:%s, mdn_loss_val:%s, affi_loss_val:%s, atom_loss_val:%s, bond_loss_val:%s"%(total_loss_val, mdn_loss_val, affi_loss_val, atom_loss_val, bond_loss_val))

	


