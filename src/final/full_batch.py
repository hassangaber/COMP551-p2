import numpy as np

# Logistic function
logistic = lambda z: 1./ (1 + np.exp(-z))

# Gradient of cost curve given inputs x,y and weights
def gradient(self,x,y,m=0,prevGrad=0):
    N, D = x.shape
    yh = logistic(np.dot(x, self.w))    # y-hat, probability class
    grad = np.dot(x.T, yh - y)/N        # divide by N because cost is mean over N points
    gradm = (m*prevGrad) + ((1-m)*grad) # calculate gradient w.r.t momentum, if m=0, then gradm=grad
    return gradm

# Logisitic regression class accounting for several input parameters for the model
class LogisticRegression:
    
    def __init__(self, add_bias=True, learning_rate=.1, epsilon=9.5e-3, max_iters=1e5, verbose=False,
                 momentum=0):
        self.add_bias = add_bias
        self.learning_rate = learning_rate
        self.epsilon = epsilon                        #to get the tolerance for the norm of gradients 
        self.max_iters = max_iters                    #maximum number of iteration of gradient descent
        self.verbose = verbose
        self.momentum = momentum


    # This method fits a logisitic regression to the training data using
    # gradient descent. The weights are calculated and ouputted.
    def fit(self,x,y):
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x,np.ones(N)])
        N,D = x.shape
        self.w = np.zeros(D)
        g = np.inf 
        t = 0
  
        while np.linalg.norm(g) > self.epsilon and t < self.max_iters:
            if (t == 0):
                g = self.gradient(x, y)
            else:
                g = self.gradient(x,y,self.momentum,g)   
            self.w = self.w - self.learning_rate * g 
            t += 1
        
        return self

    # This applies the learned model in the fit method to predict the
    # output of a given set of inputs. Returns a class of probabilities
    # for each label.
    def predict(self, x):
        if x.ndim == 1:
            x = x[:, None]
        Nt = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(Nt)])
        yh = logistic(np.dot(x,self.w))            
        return yh

# Initialize class method
LogisticRegression.gradient = gradient
