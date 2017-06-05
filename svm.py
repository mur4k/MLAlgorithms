from __future__ import division,print_function
import numpy as np
from regression import Regression
from scipy import optimize

class SVM (Regression):
	"""
	Basic class for implementing SVM models
	"""
	def __init__(self,C=1,kernel="linear",sigma=None,tol=None,maxiter=None):
		self.kernel=kernel
		self.sigma=sigma
		self.C=C
		self.tol=tol
		self.maxiter=maxiter
		self.theta=np.array([],dtype='float')
		self.landmarks=np.array([],dtype='float')

	def cost1(selx,x):
		if x>=1: return 0
		else: return -x/2+1

	def cost0(self,x):
		if x<=-1: return 0
		else: return x/2-1		

	def gaussian_kernel(self,X,y,sigma):
		return np.exp(-np.linalg.norm(X-y,axis=1)**2/(2*sigma**2))

	def linear_kernel(self,x,y):
		return np.dot(x,y)

	def similarity_matrix(self,X):
		landmarks=np.zeros((X.shape[0],self.X.shape[0]))
		for i in range(len(X)):
			landmarks[i]=self.gaussian_kernel(self.X,X[i],self.sigma)
		return landmarks

	def cost_function(self,theta,X,y):
		if self.kernel=="linear":
			X_=Regression._add_intercept(X)
			J=self.C*(np.vectorize(self.cost1)(np.dot(X_,theta)).dot(y)+\
				np.vectorize(self.cost0)(np.dot(X_,theta)).dot(np.ones(y.shape)-y))+\
				1/2*sum(self.theta[1:]**2)
		elif self.kernel=="gaussian":
			lm_=Regression._add_intercept(self.landmarks)
			J=self.C*(np.vectorize(self.cost1)(np.dot(lm_,theta)).dot(y)+\
				np.vectorize(self.cost0)(np.dot(lm_,theta)).dot(np.ones(y.shape)-y))+\
				1/2*sum(self.theta[1:]**2)
		return J

	def fit(self,X,y):
		self.X=X
		self.y=y
		if self.kernel=="linear":
			X_=Regression._add_intercept(X)
			theta=2/(X_.shape[1])*np.random.random_sample((X_.shape[1],1))-1/(X_.shape[1]) #uniform distribution at (-1/(n+1);1/(n+1)
			res=optimize.minimize(self.cost_function,theta,args=(X,y),tol=self.tol,options={'maxiter':self.maxiter})
			self.theta=res.x
		elif self.kernel=="gaussian":
			theta=2/(X.shape[0]+1)*np.random.random_sample((X.shape[0]+1,1))-1/(X.shape[0]+1) #uniform distribution at (-1/(m+1);1/(m+1))	
			self.landmarks=self.similarity_matrix(X)
			res=optimize.minimize(self.cost_function,theta,args=(X,y),tol=self.tol,options={'maxiter':self.maxiter})
			self.theta=res.x
		return self.theta, res.fun

	def predict(self,X):
		if self.kernel=="linear":
			X_=Regression._add_intercept(X)
			return (np.dot(X_,self.theta)>=0).astype(int)
		elif self.kernel=="gaussian":
			landmarks=Regression._add_intercept(self.similarity_matrix(X))
			return (np.dot(landmarks,self.theta)>=0).astype(int)

	