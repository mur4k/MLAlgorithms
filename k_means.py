from __future__ import division
import numpy as np 

class KMeans(object):
	"""
	Basic class for implementing k-means unsupervised learning algorithm for clusterization
	"""
	def __init__(self,k=1,max_iter=10000,eps=1e-5):
		"""
		Creating a model with next parametrs:
		int k -- number of clusters
		int max_iter -- determines the number of iterations while weights would be upgrading (default is 10000)
		float eps -- determines a lower difference value between two neighbour steps (default is 1e-5)
		"""
		self.k=k
		self.max_iter=max_iter
		self.eps=eps
		self.c=np.array([],dtype="float")
		self.mu=np.array([],dtype="float")
		

	def predict(self,X):
		if self.mu.shape[0]==0:
			raise NotImplementedError()
		else:
			c=np.ones(X.shape[0],dtype="float")
			for m in range(X.shape[0]-1):
				s_m=[np.dot((X[m,:]-x),(X[m,:]-x).T) for x in mu] #distances from X[m] to mu
				c[m]=min(range(len(s_m)), key=s_m.__getitem__)
			return c

	def optimize(self, X):
		J_select=[]
		mu_select=[]
		c_select=[]
		self.c=np.ones(X.shape[0],dtype="float")
		if self.k>X.shape[0]: raise ValueError()
		for i in range(10): # Different initializations 
			J=0
			c=np.ones(X.shape[0],dtype="float")
			# A random choice of centroids
			np.random.seed(0)
			indices = np.random.choice(np.arange(X.shape[0]),self.k)
			mu=X[indices,:]
			for j in range(self.max_iter): #performing optimization
				for m in range(X.shape[0]-1):
					s_m=[np.dot((X[m,:]-x),(X[m,:]-x).T) for x in mu] #distances from X[m] to mu
					c[m]=min(range(len(s_m)), key=s_m.__getitem__)
				for k in range(self.k):
					if (mu[k] - np.mean(X[c==k,:],axis=0)).all()<self.eps:break
					else: mu[k] = np.mean(X[c==k,:],axis=0)
			for k in range(self.k):
				J+=np.sum(np.dot(X[c==k,:],mu[k].T))		
			J_select.append(J/X.shape[0])
			mu_select.append(mu)
			c_select.append(c)
		J_min=min(range(len(J_select)), key=J_select.__getitem__)
		J=J_select[J_min]
		self.mu=mu_select[J_min]
		self.c=c_select[J_min]
		return J,self.mu,self.c












