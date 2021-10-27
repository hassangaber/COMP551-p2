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
    
    def __init__(self, add_bias=True, learning_rate=.1, epsilon=9.5e-3, max_iters=1e5, verbose=False, momentum=0):
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
        # the code snippet below is for gradient descent
        while np.linalg.norm(g) > self.epsilon and t < self.max_iters:
            if (t == 0):
                g = self.gradient(x, y)
            else:
                g = self.gradient(x,y,self.momentum,g)   
            self.w = self.w - self.learning_rate * g 
            t += 1
        
        if self.verbose:
            print(f'Convergence: {t} Iterations')
            print(f'Norm of gradient: {np.linalg.norm(g)}')
           #print(f'\nWeights: {self.w}\n')
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

# Attempt at another approach
#  77             for epoch in range(1, self.epochs+1):
#  78                 # Randomly shuffle the data to maximize preformance metrics
#  79                 np.random.shuffle(Xbatch)
#  80                 np.random.shuffle(Ybatch)
#  81 
#  82                 if self.verbose:
#  83                     print(f'Epoch: {epoch}')
#  84                     print(f'Convergence: {t} Iterations')
#  85                     print(f'Norm of gradient: {np.linalg.norm(g)}')
#  86                     #print(f'\nWeights: {self.w}\n')
#  87 
#  88                 N,D = Xbatch.shape
#  89                 self.w = np.zeros(D)
#  90                 g = np.inf
#  91                 t = 0
#  93                 # Iterating over batched data
#  94                 for X, Y in BatchData(Xbatch, Ybatch, self.batchSize):
#  95                     # Gradient Descent loop
#  96                     while np.linalg.norm(g) > self.epsilon and t < self.max_iters:
#  97                         if (t == 0):
#  98                             g = self.gradient(X, Y)
#  99                         else:
# 100                             g = self.gradient(X,Y,self.momentum,g)
# 101                         self.w = self.w - self.learning_rate * g
# 102                         t += 1
# 103 
# 104                     # Predicitons for Test data and validation data
# 105                     yhT = logistic(np.dot(XPT,self.w))
# 106                     yhV = logistic(np.dot(XPV,self.w))
# 107 
# 108                     predictionT, predictionV = [],[]
# 109 
# 110                     # Decision Boundaries
# 111                     for x in np.array(yhT).ravel():
# 112                         if x < 0.5: predictionT.append(0)
# 113                         else: predictionT.append(1)
# 114 
# 115                     for x in np.array(yhV).ravel():
# 116                         if x < 0.5: predictionV.append(0)
# 117                         else: predictionV.append(1)
# 118 
# 119 
# 120                 print(f'Batch Size: {self.batchSize}    TEST accuracy: {accuracy_score(YPT,predictionT)}   VALIDATION accuracy: {accuracy_score(YPV,predictionV)}')