from __future__ import division
import numpy as np

class Regression (object):
	"""
	Basic class for implementing different regression models which can be trained with gradient descent 
	"""

	def __init__(self,learn_rate=0.01,reg_coef=0,max_iter=4000,eps=0.00001):
		"""
		Creating a model with next parametrs:
		float learn_rate -- a step for updating parametrs (default is 0.01)
		float reg_coef -- regularization parametr (default is 0.1)
		int max_iter -- determines the number of iterations while weights would be upgrading (default is 4000)
		float eps -- determines a lower difference value between two neighbour steps (default is 1e-5)
		"""
		self.learn_rate=learn_rate
		self.reg_coef=reg_coef
		self.max_iter=max_iter
		self.eps=eps
		self.theta=np.array([],dtype="float")
	
	@staticmethod
	def _add_intercept(X):
		b = np.ones([X.shape[0], 1])
		return np.concatenate([b, X], axis=1)

	def _cost_function(self,X,y,theta,reg_coef=0):
		raise NotImplementedError()

	def _loss(self,x,y,theta):
		raise NotImplementedError()

	def _predict(self,X):
		raise NotImplementedError()

	def _gradient_descent(self,X,y):
		raise NotImplementedError()

