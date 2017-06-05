from __future__ import division,print_function
import numpy as np

class NormalDistribution(object):

	def __init__(self,mean,K):
		self.mean=mean
		self.K=K
		self.n=K.shape[0]

	def probability_function(self,X):
		f=1/((2*np.pi)**(self.n/2)*np.linalg.det(self.K)**0.5)*np.exp(-0.5*np.dot(np.dot((X-self.mean).T,np.linalg.inv(self.K)),(X-self.mean)))
		return f

	def affine_transformation(self, A, b):
		return NormalDistribution(np.dot(A,mean)+b,np.dot(np.dot(A,K),A.T))