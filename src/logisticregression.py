import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sigmoid function for logistic regression
sig = lambda z: 1./ (1 + np.exp(-z))

# Cost function: represents error between predicted and expected value
# as an index. It is a measure of how good the model is
def cost(x,y,w)->float:
    # z = w.T @ x
    z = np.dot(x,w)
    # Mean of the minimized function
    J = np.mean(y*np.log1p(np.exp(-z)) + (1-y)*np.log1p(np.exp(z)))
    return J

# In order to minimize the cost function, or optimize the model
# gradient descent will be utilized
# This gradient descent function minimizes the cost function wrt weights
def grad(self,x,y):
    N = x.shape
    # Probability value at x
    yhat = sig(np.dot(x, self.w))
    # Cost of each point
    grad = np.dot(x.T, yhat - y)/N
    return grad

# Implementation of the logisitic regression method with a gd method to implement sgd
class LogisticRegression:
    
    # Constructor
    def __init__(self, add_bias=True, learning_rate=.1, epsilon=1e-4, max_iters=1e5, verbose=False):
        self.add_bias = add_bias
        self.learning_rate = learning_rate
        self.epsilon = epsilon                        
        self.max_iters = max_iters 
        
    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x,np.ones(N)])
        N,D = x.shape
        self.w = np.zeros(D)
        g = np.inf 
        t = 0
        # gradient descent
        while np.linalg.norm(g) > self.epsilon and t < self.max_iters:
            g = self.grad(x, y)
            self.w = self.w - self.learning_rate * g 
            t += 1
        
        if self.verbose:
            print(f'Iterations: {t}, Norm(grad) = {np.linalg.norm(g)}')
            print(f'Weights: {self.w}')
        return self
    
    def predict(self, x):
        if x.ndim == 1:
            x = x[:, None]
        Nt = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(Nt)])
        yh = sig(np.dot(x,self.w))     
        return yh

logisticregression.grad = grad