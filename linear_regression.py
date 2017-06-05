from __future__ import division
import numpy as np
from regression import Regression
from gradient_descent import GradientDescent


class LinearRegression(Regression, GradientDescent):
	"""
	Basic class for implementing linear regression models which can be trained with gradient descent or normal equation
	"""
	
	def __init__(self,learn_rate=0.01,reg_coef=0,max_iter=6000,eps=0.00001):
		"""
		Creating a model with next parametrs:
		float learn_rate -- a step for updating parametrs (default is 0.01)
		float reg_coef -- regularization parametr (default is 0)
		int max_iter -- determines the number of iterations while weights would be upgrading (default is 6000)
		float eps -- determines a lower difference value between two neighbour steps (default is 1e-5)
		"""
		super(LinearRegression,self).__init__(learn_rate,reg_coef,max_iter,eps)

	def cost_function(self,X,y,theta,reg_coef=0):
		m=y.shape[0]
		X_=Regression._add_intercept(X)
		tmp=np.dot(X_,theta)-y.reshape(X.shape[0],1)
		J=1/(2*m)*np.dot(tmp.T,tmp)+reg_coef*sum(theta[1:]**2)
		grad=1/m*np.dot(X_.T,tmp)
		grad[1:]=grad[1:]+reg_coef/m*theta[1:]
		return J,grad

	def predict(self,X):
		if self.theta.shape[0]==0:
			raise NotImplementedError()
		else:
			return np.dot(Regression._add_intercept(X),self.theta)

	def loss(self,x,y,theta):
		x_=Regression._add_intercept(x)
		return (np.dot(x_,theta)-y)**2

	def gradient_descent(self,X,y):
		J_select=[]
		theta_select=[]
		for init in range(10):
			J,J_prev=(0,0)
			grad=np.zeros((X.shape[1],1),dtype="float")
			theta=2/(X.shape[1]+1)*np.random.random_sample((X.shape[1]+1,1))-1/(X.shape[1]+1) #uniform distribution at (-1/n;1/n)
			for i in range(self.max_iter): #performing gradient descent
				J_prev=J
				J,grad=self.cost_function(X,y,theta,self.reg_coef)
				theta=theta-self.learn_rate*grad
				if (abs(J-J_prev)<self.eps): break
			else: J,grad=self.cost_function(X,y,theta,self.reg_coef)
			J_select.append(J)
			theta_select.append(theta)
		J=min(J_select)
		self.theta=theta_select[J_select.index(J)]
		return J,theta

	def normal_equation(self,X,y):
		X_=Regression._add_intercept(X)
		tmp=np.ones((X_.shape[1],X_.shape[1]), dtype="float")
		tmp[0][0]=0
		self.theta=np.dot(np.linalg.inv(np.dot(X_.T,X_)+self.reg_coef*tmp),np.dot(X_.T,y)).reshape(X_.shape[1],1)
		"""theta=(X.T*X+reg_coef*[0 0..0])^(-1)*X.T*y
			   				     [0 1..0]
							     [0 0..1]
		"""
		return self.theta