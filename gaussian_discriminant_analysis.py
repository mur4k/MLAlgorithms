from __future__ import division, print_function
from multivariate_norm_distribution import NormalDistribution
from pca import get_normalized, get_scaled, get_normalized_and_scaled
import numpy as np

class GaussianDiscriminantAnalysis (object):
    """The Gaussian Discriminant Analysis classifier. """
    def __init__(self):
        self.classes = None
        self.X = None
        self.y = None
        # Gaussian prob. distribution parameters
        self.parameters = []

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        Sigma=self.common_covariance_matrix(X,y)
      	for i in range(len(self.classes)):
            c = self.classes[i]
            x_where_c = X[np.where(y == c)]
            self.parameters.append(NormalDistribution(x_where_c.mean(axis=0),Sigma))
            
    def common_covariance_matrix(self,X,y):
		X_=np.ones(X.shape)
		for i in range(len(self.classes)):
			c = self.classes[i]
			X_[np.where(y==c),:]=get_normalized(X[np.where(y==c)])
			return np.dot(X_.T,X_)/X.shape[0]

    # Gaussian probability distribution
    def calculate_probability(self, means, stds, X):
        return (1.0 / (np.sqrt((2.0 * np.pi) * stds))) * np.exp(-(np.power(X - means, 2) / (2 * stds**2)))

    # Calculate the prior of class c (samples where class == c / total number of samples)
    def calculate_prior(self, c):
        # Selects the rows where the class label is c
        x_where_c = self.X[np.where(self.y == c)]
        n_class_instances = np.shape(x_where_c)[0]
        n_total_instances = np.shape(self.X)[0]
        return n_class_instances / n_total_instances

    def predict(self, X):
        # Go through list of classes
        posteriors=np.zeros((len(X),len(self.classes)),dtype="float")
        posterior=np.zeros(len(X),dtype="float")
        for i in range(len(self.classes)):
            c = self.classes[i]
            prior = self.calculate_prior(c)
            # Determine P(x|Y)*p(Y)
            for j in range(len(X)):
	            posterior = self.parameters[i].probability_function(X[j])*prior
	            posteriors[j][i]=posterior
        # Get the largest probability and return the class corresponding
        # to that probability
        predictions = np.argmax(posteriors,axis=1)
        return np.choose(predictions,self.classes)