from __future__ import division
import numpy as np
from linear_regression import LinearRegression
from regression import Regression

def sigmoid(x):
	return 1/(1+np.exp(-x))

class LogisticRegression(LinearRegression):
	"""
	Basic class for implementing logistic regression models which can be trained with gradient descent
	"""
	def __init__(self,learn_rate=0.01,reg_coef=0,max_iter=10000,eps=0.0001):
		"""
		Creating a model with next parametrs:
		float learn_rate -- a step for updating parametrs (default is 0.01)
		float reg_coef -- regularization parametr (default is 0)
		int max_iter -- determines the number of iterations while weights would be upgrading (default is 10000)
		float eps -- determines a lower difference value between two neighbour steps (default is 1e-5)
		"""
		super(LogisticRegression,self).__init__(learn_rate,reg_coef,max_iter,eps)

	def loss(self,x,y,theta):
		x_=Regression._add_intercept(x)
		return -np.log(sigmoid(np.dot(x_,theta))) if y==1 else -np.log(1-sigmoid(np.dot(x_,theta)))

	def cost_function(self,X,y,theta,reg_coef=0):
		m=y.shape[0]
		X_=Regression._add_intercept(X)
		y=y.reshape(X.shape[0],1)
		J=-1/m*(np.dot(y.T,np.log(sigmoid(np.dot(X_,theta))))+\
			np.dot((np.ones((m,1),dtype="float")-y).T,np.log(np.ones((m,1),dtype="float")-sigmoid(np.dot(X_,theta)))))+\
			reg_coef*sum(theta[1:]**2)
		grad=1/m*np.dot(X_.T,sigmoid(np.dot(X_,theta))-y)
		grad[1:]=grad[1:]+reg_coef/m*theta[1:]
		return J,grad

	def predict(self,X):
		if self.theta.shape[0]==0:
			raise NotImplementedError()
		else:
			return np.round(sigmoid(np.dot(Regression._add_intercept(X),self.theta)))

	def gradient_descent(self,X,y):
		return super(LogisticRegression,self).gradient_descent(X,y)