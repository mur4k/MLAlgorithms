from __future__ import division, print_function
import numpy as np
from pca import get_normalized, get_scaled, get_normalized_and_scaled

class NaiveBayes(object):
    """The Gaussian Naive Bayes classifier. """
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
        # Calculate the mean and variance of each feature for each class
        for i in range(len(self.classes)):
            c = self.classes[i]
            # Only select the rows where the species equals the given class
            x_where_c = X[np.where(y == c)]
            # Add the mean and variance for each feature
            self.parameters.append((x_where_c.mean(axis=0),x_where_c.std(axis=0)))
            

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

    # Classify using Bayes Rule, P(Y|X) = P(X|Y)*P(Y)/P(X)
    # P(X|Y) - Probability. Gaussian distribution (given by calculate_probability)
    # P(Y) - Prior (given by calculate_prior)
    # P(X) - Scales the posterior to the range 0 - 1 (ignored)
    # Classify the sample as the class that results in the largest P(Y|X)
    # (posterior)
    def predict(self, X):
        # Go through list of classes
        posteriors=np.zeros(len(X),dtype="float")
        for i in range(len(self.classes)):
            c = self.classes[i]
            prior = self.calculate_prior(c)
            posterior = prior
            # multiply with the additional probabilties
            # Naive assumption (independence):
            # P(x1,x2,x3|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y)
            mean, std = self.parameters[i]
            # Determine P(x|Y)
            prob = self.calculate_probability(mean, std, X)
            # Multiply with the rest
            posterior*=prob.prod(axis=1)
            # Total probability = P(Y)*P(x1|Y)*P(x2|Y)*...*P(xN|Y)
            posteriors=np.c_[posteriors,posterior]
        # Get the largest probability and return the class corresponding
        # to that probability
        predictions = np.argmax(posteriors,axis=1)
        return np.choose(predictions-1,self.classes)

