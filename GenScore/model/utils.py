import torch as th
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
from torch.distributions import Normal
from torch_scatter import scatter_add
from sklearn import metrics
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_recall_curve, auc
from scipy.stats import pearsonr, spearmanr


class EarlyStopping(object):
    """Early stop tracker
	
    Save model checkpoint when observing a performance improvement on
    the validation set and early stop if improvement has not been
    observed for a particular number of epochs.
	
    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
        If ``metric`` is not None, then mode will be determined
        automatically from that.
    patience : int
        The early stopping will happen if we do not observe performance
        improvement for ``patience`` consecutive epochs.
    filename : str or None
        Filename for storing the model checkpoint. If not specified,
        we will automatically generate a file starting with ``early_stop``
        based on the current time.
    metric : str or None
        A metric name that can be used to identify if a higher value is
        better, or vice versa. Default to None. Valid options include:
        ``'r2'``, ``'mae'``, ``'rmse'``, ``'roc_auc_score'``.
	
    Examples
    --------
    Below gives a demo for a fake training process.
	
    >>> import torch
    >>> import torch.nn as nn
    >>> from torch.nn import MSELoss
    >>> from torch.optim import Adam
    >>> from dgllife.utils import EarlyStopping
	
    >>> model = nn.Linear(1, 1)
    >>> criterion = MSELoss()
    >>> # For MSE, the lower, the better
    >>> stopper = EarlyStopping(mode='lower', filename='test.pth')
    >>> optimizer = Adam(params=model.parameters(), lr=1e-3)
	
    >>> for epoch in range(1000):
    >>>     x = torch.randn(1, 1) # Fake input
    >>>     y = torch.randn(1, 1) # Fake label
    >>>     pred = model(x)
    >>>     loss = criterion(y, pred)
    >>>     optimizer.zero_grad()
    >>>     loss.backward()
    >>>     optimizer.step()
    >>>     early_stop = stopper.step(loss.detach().data, model)
    >>>     if early_stop:
    >>>         break
	
    >>> # Load the final parameters saved by the model
    >>> stopper.load_checkpoint(model)
    """
    def __init__(self, mode='higher', patience=10, filename=None, metric=None):
        if filename is None:
            #dt = datetime.datetime.now()
            filename = 'early_stop.pth'
		
        if metric is not None:
            assert metric in ['rp', 'rs', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score'], \
                "Expect metric to be 'rp' or 'rs' or 'mae' or " \
                "'rmse' or 'roc_auc_score', got {}".format(metric)
            if metric in ['rp', 'rs', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'
		
        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.timestep = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False
	
    def _check_higher(self, score, prev_best_score):
        """Check if the new score is higher than the previous best score.
	
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
	
        Returns
        -------
        bool
            Whether the new score is higher than the previous best score.
        """
        return score > prev_best_score
	
    def _check_lower(self, score, prev_best_score):
        """Check if the new score is lower than the previous best score.
	
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.

        Returns
        -------
        bool
            Whether the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score, model):
        """Update based on a new score.
	
        The new score is typically model performance on the validation set
        for a new epoch.

        Parameters
        ----------
        score : float
            New score.
        model : nn.Module
            Model instance.
	
        Returns
        -------
        bool
            Whether an early stop should be performed.
        """
        self.timestep += 1
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
	
    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.
	
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        th.save({'model_state_dict': model.state_dict(),
                    'timestep': self.timestep}, self.filename)
	
    def load_checkpoint(self, model):
        '''Load the latest checkpoint
	
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        model.load_state_dict(th.load(self.filename)['model_state_dict'])



def run_a_train_epoch(epoch, model, data_loader, optimizer, mdn_weight=1.0, affi_weight=1.0, aux_weight=0.001, dist_threhold=None, device='cpu'):
	model.train()
	total_loss = 0
	mdn_loss = 0
	affi_loss = 0
	atom_loss = 0
	bond_loss = 0
	probs = []
	for batch_id, batch_data in enumerate(data_loader):
		pdbids, bgp, bgl, labels = batch_data
		bgl, bgp = bgl.to(device), bgp.to(device)
		
		atom_labels = th.argmax(bgl.x[:,:17], dim=1, keepdim=False)
		bond_labels = th.argmax(bgl.edge_attr[:,:4], dim=1, keepdim=False)
		
		pi, sigma, mu, dist, atom_types, bond_types, batch = model(bgl, bgp)
	
		mdn, prob = mdn_loss_fn(pi, sigma, mu, dist)
		mdn = mdn[th.where(dist <= model.dist_threhold)[0]]
		mdn = mdn.mean()
		
		
		batch = batch.to(device)
		if dist_threhold is not None:
			prob = prob[th.where(dist <= dist_threhold)[0]] 
			y = scatter_add(prob, batch[th.where(dist <= dist_threhold)[0]] , dim=0, dim_size=batch.unique().size(0))
		else:
			y = scatter_add(prob, batch, dim=0, dim_size=batch.unique().size(0))
		
		labels = labels.float().type_as(y).to(device)
		
		affi = th.corrcoef(th.stack([y, labels]))[1,0]	
		atom = F.cross_entropy(atom_types, atom_labels)
		bond = F.cross_entropy(bond_types, bond_labels)
		loss = (mdn * mdn_weight) + (affi * affi_weight) + (atom * aux_weight) + (bond * aux_weight)
		probs.append(y)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		
		#total_loss += loss.item() * batch.unique().size(0)
		mdn_loss += mdn.item() * batch.unique().size(0)
		#affi_loss += affi.item() * batch.unique().size(0)
		atom_loss += atom.item() * batch.unique().size(0)
		bond_loss += bond.item() * batch.unique().size(0)
		total_loss += mdn_loss + atom_loss * aux_weight + bond_loss * aux_weight
		
		#print('Step, Total Loss: {:.3f}, MDN: {:.3f}'.format(total_loss, mdn_loss))
		if np.isinf(mdn_loss) or np.isnan(mdn_loss): break
		del bgl, bgp, atom_labels, bond_labels, pi, sigma, mu, dist, atom_types, bond_types, batch, mdn, atom, bond, affi, loss, prob, y
		th.cuda.empty_cache()
	
	ys = th.cat(probs)
	affi_loss = th.corrcoef(th.stack([ys, data_loader.dataset.labels.to(device)]))[1,0].item()
		
	return total_loss / len(data_loader.dataset)+ affi_loss* affi_weight, mdn_loss / len(data_loader.dataset), affi_loss, atom_loss / len(data_loader.dataset), bond_loss / len(data_loader.dataset)



def run_an_eval_epoch(model, data_loader, pred=False, atom_contribution=False, res_contribution=False, dist_threhold=None, mdn_weight=1.0, affi_weight=1.0, aux_weight=0.001, device='cpu'):
	model.eval()
	total_loss = 0
	affi_loss = 0
	mdn_loss = 0
	atom_loss = 0
	bond_loss = 0
	probs = []
	at_contrs = []
	res_contrs = []
	with th.no_grad():
		for batch_id, batch_data in enumerate(data_loader):
			pdbids, bgp, bgl, labels = batch_data
			bgl, bgp = bgl.to(device), bgp.to(device)
			atom_labels = th.argmax(bgl.x[:,:17], dim=1, keepdim=False)
			bond_labels = th.argmax(bgl.edge_attr[:,:4], dim=1, keepdim=False)
			
			pi, sigma, mu, dist, atom_types, bond_types, batch = model(bgl, bgp)
			
			if pred or atom_contribution or res_contribution:
				prob = calculate_probablity(pi, sigma, mu, dist)
				if dist_threhold is not None:
					prob[th.where(dist > dist_threhold)[0]] = 0.
				
				batch = batch.to(device)
				if pred:
					probx = scatter_add(prob, batch, dim=0, dim_size=batch.unique().size(0))
					probs.append(probx)
				if atom_contribution or res_contribution:				
					contribs = [prob[batch==i].reshape(len(bgl.x[bgl.batch==i]), len(bgp.x[bgp.batch==i])) for i in th.arange(0, len(batch.unique()))]
					if atom_contribution:
						at_contrs.extend([contribs[i].sum(1).cpu().detach().numpy() for i in th.arange(0, len(batch.unique()))])
					if res_contribution:
						res_contrs.extend([contribs[i].sum(0).cpu().detach().numpy() for i in th.arange(0, len(batch.unique()))])
			
			else:
				mdn, prob = mdn_loss_fn(pi, sigma, mu, dist)
				mdn = mdn[th.where(dist <= model.dist_threhold)[0]]
				mdn = mdn.mean()
				
				batch = batch.to(device)	
				if dist_threhold is not None:
					prob = prob[th.where(dist <= dist_threhold)[0]] 
					y = scatter_add(prob, batch[th.where(dist <= dist_threhold)[0]] , dim=0, dim_size=batch.unique().size(0))
				else:	
					y = scatter_add(prob, batch[th.where(dist <= dist_threhold)[0]] , dim=0, dim_size=batch.unique().size(0))
				#labels = labels.float().type_as(y).to(device)
				
				#affi = F.mse_loss(y, labels)	
				
				atom = F.cross_entropy(atom_types, atom_labels)
				bond = F.cross_entropy(bond_types, bond_labels)
				loss = (mdn * mdn_weight) + (atom * aux_weight) + (bond * aux_weight)
				#loss = mdn + affi * affi_weight + (atom * aux_weight) + (bond * aux_weight)
				
				probs.append(y)
				
				total_loss += loss.item() * batch.unique().size(0)
				mdn_loss += mdn.item() * batch.unique().size(0)
				atom_loss += atom.item() * batch.unique().size(0)
				bond_loss += bond.item() * batch.unique().size(0)			
			
			del bgl, bgp, atom_labels, bond_labels, pi, sigma, mu, dist, atom_types, bond_types, batch 
			th.cuda.empty_cache()
	
	if atom_contribution or res_contribution:
		if pred:
			preds = th.cat(probs)
			return [preds.cpu().detach().numpy(),at_contrs,res_contrs]
		else:
			return [None, at_contrs,res_contrs]
	else:
		if pred:
			preds = th.cat(probs)
			return preds.cpu().detach().numpy()
		else:		
			ys = th.cat(probs)
			affi_loss = th.corrcoef(th.stack([ys, data_loader.dataset.labels.to(device)]))[1,0].item()
			#total_loss += - affi_loss 
			del ys
			return total_loss / len(data_loader.dataset)+ affi_loss * affi_weight, mdn_loss / len(data_loader.dataset), affi_loss, atom_loss / len(data_loader.dataset), bond_loss / len(data_loader.dataset)


def set_random_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    #th.backends.cudnn.benchmark = False
    #th.backends.cudnn.deterministic = True
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)



def calculate_probablity(pi, sigma, mu, y):
    normal = Normal(mu, sigma)
    logprob = normal.log_prob(y.expand_as(normal.loc))
    logprob += th.log(pi)
    prob = logprob.exp().sum(1)
	
    return prob



def mdn_loss_fn(pi, sigma, mu, y, eps1=1e-10, eps2=1e-10):
    normal = Normal(mu, sigma)
    #loss = th.exp(normal.log_prob(y.expand_as(normal.loc)))
    #loss = th.sum(loss * pi, dim=1)
    #loss = -th.log(loss)
    loglik = normal.log_prob(y.expand_as(normal.loc))
    #loss = -th.logsumexp(th.log(pi + eps) + loglik, dim=1)
    prob = (th.log(pi + eps1) + loglik).exp().sum(1)
    loss = -th.log(prob + eps2)
    return loss, prob
