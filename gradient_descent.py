from __future__ import division
import numpy as np

class GradientDescent(object):
	"""
	Basic class for implementing gradient descent
	"""

	def __init__(self, func, grad, learn_rate=0.01, eps=1e-5, max_iter=6000):
		"""
		func=callable function that takes np.array and returns np.array;
		grad=callable gradient to your function that takes np.array and returns np.array
		float learn_rate -- a step for updating parametrs (default is 0.01)
		float eps -- determines a lower difference value between two neighbour steps (default is 1e-5)
		int max_iter -- determines the number of iterations while weights would be upgrading (default is 4000)
		"""
		self.func = np.vectorize(func)
		self.grad = grad
		self.max_iter = max_iter
		self.learn_rate = learn_rate
		self.eps = eps
		self.x_min = None

	def minimize(self, *initial_guess):
		x = initial_guess
		f_hist = []
		f_hist.append(self.func(*x))
		for i in range(self.max_iter):
			x_prev = x
			x = x - self.learn_rate*self.grad(*x)
			f_hist.append(self.func(*x))
			if (np.abs(x_prev-x) < self.eps).all(): break
		self.x_min = x
		return (x,f_hist)