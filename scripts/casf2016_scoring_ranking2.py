import numpy as np
import torch as th
from joblib import Parallel, delayed
import pandas as pd
import os, sys
import pickle
from scipy.stats import pearsonr
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
sys.path.append("/home/shenchao/rtmscorepyg/code")
from torch_geometric.loader import DataLoader
from GenScore.data.data import PDBbindDataset
from GenScore.model.model import GenScore, GraphTransformer, GatedGCN
from GenScore.model.utils import run_an_eval_epoch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
args={}
args["batch_size"] = 128
args["dist_threhold"] = 5.
args['device'] = 'cuda' if th.cuda.is_available() else 'cpu'
args['seeds'] = 126
args["num_workers"] = 10
args["model_path"] = "/home/shenchao/rtmscorepyg/code/trained_models/GatedGCN_ft_1.0_1.pth"
args["data_dir"] = "/home/shenchao/rtmscorepyg/dataset/rtmscore_s"
args["test_prefix"] = "v2020_casf"
args["cutoff"] = 10.0
args["encoder"] = "gatedgcn"
args["num_node_featsp"] = 41
args["num_node_featsl"] = 41
args["num_edge_featsp"] = 5
args["num_edge_featsl"] = 10
args["hidden_dim0"] = 128 
args["hidden_dim"] = 128
args["n_gaussians"] = 10
args["dropout_rate"] = 0.15
args["outprefix"] = "gatedgcn1x5"


def scoring(ids, prots, ligs, modpath,
			cut=10.0,
			explicit_H=False, 
			use_chirality=True,
			parallel=False,
			**kwargs
			):
	"""
	prot: The input protein file ('.pdb')
	lig: The input ligand file ('.sdf|.mol2', multiple ligands are supported)
	modpath: The path to store the pre-trained model
	gen_pocket: whether to generate the pocket from the protein file.
	reflig: The reference ligand to determine the pocket.
	cutoff: The distance within the reference ligand to determine the pocket.
	explicit_H: whether to use explicit hydrogen atoms to represent the molecules.
	use_chirality: whether to adopt the information of chirality to represent the molecules.	
	parallel: whether to generate the graphs in parallel. (This argument is suitable for the situations when there are lots of ligands/poses)
	kwargs: other arguments related with model
	"""
	#try:
	data = PDBbindDataset(ids=ids,prots=prots,ligs=ligs)
						
	test_loader = DataLoader(dataset=data, 
							batch_size=kwargs["batch_size"],
							shuffle=False, 
							num_workers=kwargs["num_workers"])
	
	if kwargs["encoder"] == "gt":
		ligmodel = GraphTransformer(in_channels=kwargs["num_node_featsl"], 
									edge_features=kwargs["num_edge_featsl"], 
									num_hidden_channels=kwargs["hidden_dim0"],
									activ_fn=th.nn.SiLU(),
									transformer_residual=True,
									num_attention_heads=4,
									norm_to_apply='batch',
									dropout_rate=0.15,
									num_layers=6
									)
		
		protmodel = GraphTransformer(in_channels=kwargs["num_node_featsp"], 
									edge_features=kwargs["num_edge_featsp"], 
									num_hidden_channels=kwargs["hidden_dim0"],
									activ_fn=th.nn.SiLU(),
									transformer_residual=True,
									num_attention_heads=4,
									norm_to_apply='batch',
									dropout_rate=0.15,
									num_layers=6
									)
	elif kwargs["encoder"] == "gatedgcn":
		ligmodel = GatedGCN(in_channels=kwargs["num_node_featsl"], 
							edge_features=kwargs["num_edge_featsl"], 
							num_hidden_channels=kwargs["hidden_dim0"], 
							residual=True,
							dropout_rate=0.15,
							equivstable_pe=False,
							num_layers=6
							)
		
		protmodel = GatedGCN(in_channels=kwargs["num_node_featsp"], 
							edge_features=kwargs["num_edge_featsp"], 
							num_hidden_channels=kwargs["hidden_dim0"], 
							residual=True,
							dropout_rate=0.15,
							equivstable_pe=False,
							num_layers=6
							)
	else:
		raise ValueError("encoder should be \"gt\" or \"gatedgcn\"!")
						
	model = GenScore(ligmodel, protmodel, 
					in_channels=kwargs["hidden_dim0"], 
					hidden_dim=kwargs["hidden_dim"], 
					n_gaussians=kwargs["n_gaussians"], 
					dropout_rate=kwargs["dropout_rate"], 
					dist_threhold=kwargs["dist_threhold"]).to(kwargs['device'])
	
	checkpoint = th.load(modpath, map_location=th.device(kwargs['device']))
	model.load_state_dict(checkpoint['model_state_dict']) 
	preds = run_an_eval_epoch(model, test_loader, pred=True, dist_threhold=kwargs['dist_threhold'], device=kwargs['device'])	
	return data.pdbids, preds
	#except:
	#	print("failed to scoring for {} and {}".format(prot, lig))
	#	return None, None

def obtain_metrics(df):
	#Calculate the Pearson correlation coefficient
	regr = linear_model.LinearRegression()
	regr.fit(df.score.values.reshape(-1,1), df.logKa.values.reshape(-1,1))
	preds = regr.predict(df.score.values.reshape(-1,1))
	rp = pearsonr(df.logKa, df.score)[0]
	#rp = df[["logKa","score"]].corr().iloc[0,1]
	mse = mean_squared_error(df.logKa, preds)
	num = df.shape[0]
	sd = np.sqrt((mse*num)/(num-1))
	#return rp, sd, num
	print("The regression equation: logKa = %.2f + %.2f * Score"%(float(regr.coef_), float(regr.intercept_)))
	print("Number of favorable sample (N): %d"%num)
	print("Pearson correlation coefficient (R): %.3f"%rp)
	print("Standard deviation in fitting (SD): %.2f"%sd)


prots='%s/%s_prot.pt'%(args["data_dir"],args["test_prefix"])
ligs='%s/%s_lig.pt'%(args["data_dir"],args["test_prefix"])
ids='%s/%s_ids.npy'%(args["data_dir"],args["test_prefix"])

#data = VSDataset(ids=ids,graphs=graphs)

_, preds = scoring(ids, prots, ligs, args["model_path"], parallel=True, **args)

df = pd.read_csv("/home/shenchao/test/CASF-2016/power_scoring/CoreSet.dat", sep='[,,\t, ]+', header=0, engine='python')
df_score = pd.DataFrame(zip(np.load(ids)[0],preds), columns=["#code","score"])
testdf = pd.merge(df,df_score,on='#code')
testdf[["#code","score"]].to_csv("/home/shenchao/test/CASF-2016/power_ranking/examples/%s.dat"%args["outprefix"], index=False, sep="\t")

obtain_metrics(testdf)






