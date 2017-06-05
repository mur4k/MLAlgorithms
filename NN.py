from __future__ import division
import numpy as np
from regression import Regression
from scipy import optimize

def sigmoid(x):
    return 1/(1+np.exp(-x.astype("float64")))

class NeuralNetwork(Regression):

    def __init__(self,layers,learn_rate=0.01,reg_coef=0,max_iter=25000,eps=0.0001):
        self.l=len(layers)
        self.layers=layers
        self.W=[]
        for i in range(self.l-1):
            self.W.append(np.random.randn(layers[i+1],layers[i]+1).astype("float64"))
        super(NeuralNetwork,self).__init__(learn_rate,reg_coef,max_iter,eps)
        # print([x.shape for x in self.W])

    def from_weights_to_vector(self):
        theta=np.array([],dtype="float64")
        for i in range(self.l-1):
            theta=np.hstack((theta,self.W[i].flatten()))
        return theta

    def from_vector_to_weights(self,theta):
        W=[]
        prev_size=0
        # print(theta.shape)
        for i in range(self.l-1):
            W.append(np.array([],dtype="float64"))
            W[i]=theta[prev_size:(prev_size+self.layers[i+1]*(self.layers[i]+1))]
            # print(W[i].shape)
            W[i]=W[i].reshape((self.layers[i+1],self.layers[i]+1)).astype("float64")
            prev_size+=self.layers[i+1]*self.layers[i]+1
        return W

    def sum_of_unregularized_theta(self,theta=None,W=None):
        s=0
        prev_size=0
        if theta==None:
            for i in range(self.l-1):
                s+=sum(W[i][:,0]**2)
        elif W==None:
            for i in range(self.l-1):
                s+=sum(theta[prev_size:\
                            (prev_size+self.layers[i+1]*(self.layers[i]+1)):\
                            self.layers[i]+1]**2)
        return s
        
    def cost_function(self,theta,X,y):
        m=X.shape[0]
        W=self.from_vector_to_weights(theta)
        a=self.fprop(X,W)
        J=-1/m*sum(np.sum(np.log(a[self.l-1])*y,axis=1)+np.sum(np.log(1-a[self.l-1])*(1-y),axis=1))+\
            self.reg_coef/(2*m)*(sum(theta**2)-self.sum_of_unregularized_theta(theta=theta,W=None))
        return J

    def fprop(self, X, W):
        # Forward prop
        a=[0]
        a[0]=np.array(Regression._add_intercept(X),dtype="float64")
        for i in range(1,self.l):
            z=np.array([],dtype="float64")
            z=a[i-1].dot(W[i-1].T)
            g_z=sigmoid(z)
            a.append(np.array(Regression._add_intercept(g_z),dtype="float64"))
        a[self.l-1]=a[self.l-1][:,1:]
        # print([x.shape for x in a])
        return a
        

    def predict(self, X):
        a=self.fprop(X,self.W)
        # Get max prob. for class
        return np.argmax(a[self.l-1],axis=1)

    def bprop(self, X, y):
        # Back prop
        m=X.shape[0]
        a=self.fprop(X,self.W)
        err=[a[self.l-1]-y]
        err.insert(0,err[0].dot(self.W[self.l-2]).astype("float64"))
        err[0][:,1:]*=a[self.l-2][:,1:]*(1-a[self.l-2][:,1:])
        dW=[]
        for i in range(self.l-3,-1,-1):
            err.insert(0,err[0][:,1:].dot(self.W[i]).astype("float64"))
            err[0][:,1:]*=a[i][:,1:]*(1-a[i][:,1:])
        # print([x.shape for x in err])
        for i in range(self.l-2):
            dW.append(err[i+1][:,1:].T.dot(a[i]).astype("float64")/m)
            dW[i][:,1:]+=self.reg_coef*self.W[i][:,1:]
        dW.append(err[self.l-1].T.dot(a[self.l-2]).astype("float64")/m)
        # print([x.shape for x in dW])
        return dW

    def gradient_descent(self,X,y):
        J,J_prev=(0,0)
        theta=self.from_weights_to_vector()
        for i in range(self.max_iter): #performing gradient descent
            dW=self.bprop(X,y)
            J_prev=J
            J=self.cost_function(theta,X,y)
            for i in range(len(self.W)):
                self.W[i]-=self.learn_rate*dW[i]
            #if (abs(J-J_prev)<self.eps): break
        else: J=self.cost_function(theta,X,y)
        return self.W,J

    def train(self, X, y):
        # Train for each iteration
        theta=self.from_weights_to_vector().astype("float64")
        res=optimize.minimize(self.cost_function,theta,args=(X,y),method='L-BFGS-B',tol=self.eps,options={'maxiter':self.max_iter})
        self.W=self.from_vector_to_weights(res.x)
        return self.W,res.fun
        