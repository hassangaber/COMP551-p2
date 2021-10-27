import numpy as np

logistic = lambda z: 1./ (1 + np.exp(-z))       #logistic function

# This is the gradient given some inputs x,y for a logistic regression
def gradient_Momentum(self, x, y, m = 0, old_grad = 0):
    N, D = x.shape
    yh = logistic(np.dot(x, self.w))    # Preidictions, sigma(w.T @ x)
    grad = np.dot(x.T, yh - y)/N        # divide by N because cost is mean over N points
    mom = (m*old_grad) + ((1-m)*grad)
    return mom                         

# This class explicitly runs a logisitc regression on the data and
# optimizes the cost function using gradient descent
class LogisticRegression:
    
    def __init__(self, add_bias=True, learning_rate=.1, epsilon=1e-3, max_iters=1e6, verbose=False, momentum = 0):
        self.add_bias = add_bias
        self.learning_rate = learning_rate
        self.epsilon = epsilon                        #to get the tolerance for the norm of gradients 
        self.max_iters = max_iters                    #maximum number of iteration of gradient descent
        self.verbose = verbose
        self.momentum = momentum
        
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
        # the code snippet below is for gradient descent
        while np.linalg.norm(g) > self.epsilon and t < self.max_iters:
            if (t == 0):
                g = self.gradient_Momentum(x, y)
            else:
                g = self.gradient_Momentum(x,y,self.momentum, g)   
            self.w = self.w - self.learning_rate * g 
            t += 1
        
        if self.verbose:
            print(f'Convergence: {t} Iterations')
            print(f'Norm of gradient: {np.linalg.norm(g)}')
            #print(f'\nWeights: {self.w}\n')
        return self
    
    def predict(self, x):
        if x.ndim == 1:
            x = x[:, None]
        Nt = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(Nt)])
        yh = logistic(np.dot(x,self.w))            #predict output
        return yh

LogisticRegression.gradient_Momentum = gradient_Momentum  
