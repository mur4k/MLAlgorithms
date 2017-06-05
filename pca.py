from __future__ import division
import numpy as np

def get_normalized_and_scaled(X):
	return (X - np.mean(X,axis=0))/np.std(X,axis=0)

def get_normalized(X):
	return (X - np.mean(X,axis=0))

def get_scaled(X):
	return X/np.std(X,axis=0)

class PCA(object):
	"""
	Basic class for implementing principal component analysis
	"""
	def __init__(self, n_comp):
		"""
		Creating a model with next parametrs:
		int n_comp -- dimensionality for reduction
		"""
		self.U_reduce=np.array([],dtype="float")
		self.n_comp=n_comp

	def optimize_n_comp(self,X):
		n_comp_select=[]
		var_loss_select=[]
		X_cov=self.covariance_matrix(X)
		U,s,V=np.linalg.svd(X_cov)
		for k in range(1,X.shape[1]):
			var_loss=1-np.sum(s[:k])/np.sum(s)
			if ((var_loss)<0.01):
				var_loss_select.append(var_loss)
				n_comp_select.append(k)
		i_min=min(range(len(var_loss_select)), key=var_loss_select.__getitem__)
		self.n_comp=n_comp_select[i_min]
		return self.n_comp, var_loss_select[i_min]


	def rebuild(self,Z):
		return np.dot(Z,self.U_reduce.T)

	def covariance_matrix(self,X):
		X_=get_normalized_and_scaled(X)
		return np.dot(X_.T,X_)/X.shape[0]

	def perform_pca(self,X):
		if self.n_comp>X.shape[1]: raise ValueError()
		X_cov=self.covariance_matrix(X)
		U,s,V=np.linalg.svd(X_cov)
		self.U_reduce=U[:,:self.n_comp]
		X_reduce=np.dot(X,self.U_reduce)
		return X_reduce
