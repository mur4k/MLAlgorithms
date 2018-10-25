import numpy as np
from graphviz import Digraph

def misclassification_rate(y):
	return np.max([1 - np.mean(y == x) for x in np.unique(y)])

def gini(y):
	return 1 - np.sum([np.mean(y == x) ** 2 for x in np.unique(y)])

def entropy(y):
	probs = np.array([np.mean(y == x) for x in np.unique(y)])
	return - np.sum(probs * np.log(probs))

def RSS(y):
	return np.sum(np.power(y - np.mean(y), 2))

def Node(parent=None, feature_value=None, feature_idx=None, is_categorical=None, split_cost=None, criterion_value=None, distribution=None, left=None, right=None):
	return dict([('feature_value', feature_value), 
				 ('feature_idx', feature_idx), 
				 ('is_categorical', is_categorical), 
				 ('criterion_value', criterion_value),
				 ('split_cost', split_cost), 
				 ('distribution', distribution),
				 ('left', left), 
				 ('right', right)])

class DecisionTree(object):
	def __init__(self, criterion='gini', max_depth=2, min_split_node=2, eps=0.01, problem_type='classification'):
		self.criterion = criterion
		self.max_depth = max_depth
		self.min_split_node = min_split_node
		self.eps = eps
		self.problem_type = problem_type
		self.depth = 0
		self.root_node = None

	def criterion_cost(x, y, split_value, is_categorical, criterion):
		return criterion(y[x <= split_value]) + criterion(y[x > split_value]) if not is_categorical \
			   else criterion(y[x == split_value]) + criterion(y[x != split_value])

	def get_criterion_value(self, y):
		if self.problem_type == 'classification':
			if self.criterion == 'gini':
				return gini(y)
			elif self.criterion == 'entropy':
				return entropy(y)
			elif self.criterion == 'misclassification_rate':
				return misclassification_rate(y)
		elif self.problem_type == 'regression' and criterion == 'RSS':
			return RSS(y)

	def get_cost_value(self, x, y, split_value, is_categorical):
		if self.problem_type == 'classification':
			if self.criterion == 'gini':
				return DecisionTree.criterion_cost(x, y, split_value, is_categorical, gini)
			elif self.criterion == 'entropy':
				return DecisionTree.criterion_cost(x, y, split_value, is_categorical, entropy)
			elif self.criterion == 'misclassification_rate':
				return DecisionTree.criterion_cost(x, y, split_value, is_categorical, misclassification_rate)
		elif self.problem_type == 'regression' and criterion == 'RSS':
			return DecisionTree.criterion_cost(x, y, split_value, RSS)

	def get_distribution(self, y):
		if self.problem_type == 'classification':
			return [np.sum(y == x) for x in self.c]
		elif self.problem_type == 'regression':
			return [y.size, np.mean(y)]

	def get_split_value(self, X, y):
		min_feature_idx = None
		min_feature_value = None
		min_split_cost = None
		is_categorical = None
		for feature_idx in self.order:
			if self.categorical_features is not None:
				if feature_idx in self.categorical_features:
					values = np.unique(X[:, feature_idx])
					for value in values:
						criterion_cost = self.get_cost_value(values, y, value, True)
						if min_split_cost is None or criterion_cost < min_split_cost:
							min_feature_idx, min_feature_value, min_split_cost, is_categorical = feature_idx, value, criterion_cost, True
			else:
				indicies = np.argsort(X[:, feature_idx])
				values = X[:, feature_idx][indicies]
				y_values = y[indicies]
				switch_indicies = np.where(y_values[:-1] != y_values[1:])[0]
				threshold_values = (values[switch_indicies] + values[switch_indicies + 1]) / 2
				for value in threshold_values:
					criterion_cost = self.get_cost_value(values, y_values, value, False)
					if min_split_cost is None or criterion_cost < min_split_cost:
						min_feature_idx, min_feature_value, min_split_cost, is_categorical = feature_idx, value, criterion_cost, False
		return (min_feature_idx, min_feature_value, is_categorical, min_split_cost) 

	def worth_splitting(self, node, parent_criterion_value, current_depth):
		return node['criterion_value'] > self.eps and \
			   current_depth + 1 <= self.max_depth and \
			   self.min_split_node <= np.array(node['distribution']).sum() if self.problem_type == 'classification' else node['distribution'][0]

	def fit_tree(self, X, y, parent_criterion_value, current_depth):
		node = Node()
		node['distribution'] = self.get_distribution(y)
		node['criterion_value'] = self.get_criterion_value(y)
		if self.worth_splitting(node, parent_criterion_value, current_depth):
			if current_depth + 1 >= self.depth:
				self.depth += 1
			min_feature_idx, min_feature_value, is_categorical, min_split_cost = self.get_split_value(X, y)
			node['feature_idx'] = min_feature_idx
			node['feature_value'] = min_feature_value
			node['split_cost'] = min_split_cost
			node['is_categorical'] = is_categorical
			if is_categorical:
				node['left'] = self.fit_tree(X[X[:, min_feature_idx] == node['feature_value']], 
											 y[X[:, min_feature_idx] == node['feature_value']], 
											 node['criterion_value'], 
											 current_depth + 1) 
				node['right'] = self.fit_tree(X[X[:, min_feature_idx] != node['feature_value']], 
										      y[X[:, min_feature_idx] != node['feature_value']], 
										      node['criterion_value'], 
										      current_depth + 1)
				return node
			else:
				node['left'] = self.fit_tree(X[X[:, min_feature_idx] <= node['feature_value']], 
											 y[X[:, min_feature_idx] <= node['feature_value']], 
											 node['criterion_value'], 
											 current_depth + 1) 
				node['right'] = self.fit_tree(X[X[:, min_feature_idx] > node['feature_value']], 
											  y[X[:, min_feature_idx] > node['feature_value']], 
											  node['criterion_value'], 
											  current_depth + 1)
				return node
		else:
			return node		

	def fit(self, X_train, y_train, categorical_features=None):
		del self.root_node
		self.depth = 0
		self.categorical_features = categorical_features
		self.order = np.arange(X_train.shape[1])
		np.random.shuffle(self.order)
		if self.problem_type == 'classification':
			self.c = np.unique(y_train)
		self.root_node = self.fit_tree(X_train, y_train, 0., 0)

	def predict(self, X_pred):
		y_pred = np.zeros(X_pred.shape[0])
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
			node = self.root_node
			while not (node['left'] is None and node['right'] is None):
				if node['is_categorical']:
					if X_pred[i, node['feature_idx']] == node['feature_value']:
						node = node['left']
					else:
						node = node['right']
				else:
					if X_pred[i, node['feature_idx']] <= node['feature_value']:
						node = node['left']
					else:
						node = node['right']
			if self.problem_type == 'classification':
				y_pred[i] = node['distribution'] / np.sum(node['distribution'])
			elif self.problem_type == 'regression':
				y_pred[i] = node['distribution'][1]
		return y_pred

	def visualize_node(self, node, node_number):
		if not (node['left'] is None and node['right'] is None):
			self.g.node(str(int(node_number)), 
						label=r'feature #{} {} {} \n criterion value = {} \n class distr = {}'.\
								format(node['feature_idx'], 
		   	   	    				   '<=' if self.problem_type == 'classification' else '!=',
		   							   node['feature_value'],
		   							   node['criterion_value'],
		   							   node['distribution']))		
			self.visualize_node(node['left'], 2 * node_number + 1)
			self.visualize_node(node['right'], 2 * node_number + 2)
			if node_number != 0:
				self.g.edge(str((node_number - 1) // 2), str(int(node_number)))
		else:
			self.g.node(str(int(node_number)), 
						label=r'criterion value = {} \n class distr = {}'.\
								format(node['criterion_value'], 
									   node['distribution']))
			self.g.edge(str((node_number - 1) // 2), str(int(node_number)))

	def export_to_graphviz(self, filename=None):
		self.g = Digraph('DecisionTree')
		node = self.root_node
		self.visualize_node(node, 0)
		self.g.view()