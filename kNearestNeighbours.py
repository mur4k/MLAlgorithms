import numpy as np

class kNearestNeighbours(object):
	def __init__(self, X_train, y_train, k=1, distance='euclidean', weighted=False, sigma=None, problem_type='classification'):
		self.X_train = X_train
		self.y_train = y_train
		self.k = k
		self.distance = distance
		self.weighted = weighted
		self.sigma = sigma
		self.problem_type = problem_type
		if self.problem_type == 'classification':
			self.c = np.unique(y_train)

	#malahnobius distance	
	def malahnobius_distance(u, v, sigma):
		return np.sqrt(np.dot(np.dot((u - v), np.linalg.inv(sigma)), (u - v).T))

	#l2 distance
	def euclidean_distance(u, v):
		return kNearestNeighbours.malahnobius_distance(u, v, np.eye(u.shape[0]))

	#l1 distance 
	def manhattan_distance(u, v):
		return np.sum(np.abs(u - v))

	def calculate_distances(self, x):
		distances = np.zeros(self.X_train.shape[0])
		for j in range(distances.size):
			if self.distance == 'euclidean':
				distances[j] = kNearestNeighbours.euclidean_distance(x, self.X_train[j, :])
			elif self.distance == 'manhattan':
				distances[j] = kNearestNeighbours.manhattan_distance(x, self.X_train[j, :])
			elif self.distance == 'malahnobius':
				distances[j] = kNearestNeighbours.malahnobius_distance(x, self.X_train[j, :], self.sigma)
		return distances

	def calculate_proba(self, weights, k_min_indices):
		if self.problem_type == 'classification':
			y_pred = list(map(lambda val: np.sum(weights[self.y_train[k_min_indices] == val]), self.c))
		elif self.problem_type == 'regression':
			y_pred = np.sum(weights * self.y_train[k_min_indices])
		return y_pred

	def predict(self, X_pred):
		y_pred = np.zeros(X_pred.shape[0], dtype=self.y_train.dtype)
		if self.problem_type == 'classification':
			y_pred_proba = self.predict_proba(X_pred)
			y_pred = self.c[np.argmax(y_pred_proba, axis=1)]
		elif self.problem_type == 'regression':
			y_pred = self.predict_proba(X_pred)
		return y_pred

	def predict_proba(self, X_pred):
		if self.problem_type == 'classification':
			y_pred = np.zeros((X_pred.shape[0], self.c.size))
		elif self.problem_type == 'regression':
			y_pred = np.zeros(X_pred.shape[0])
		for i in range(y_pred.shape[0]):
			distances = self.calculate_distances(X_pred[i, :])
			k_min_indices = np.argpartition(distances, self.k)[:self.k] 
			if self.weighted:
				weights = 1 - distances[k_min_indices] / np.sum(distances[k_min_indices])
				y_pred[i] = self.calculate_proba(weights, k_min_indices)
			else:
				weights = np.ones(self.k, dtype=np.float64) / self.k
				y_pred[i] = self.calculate_proba(weights, k_min_indices)
		return y_pred