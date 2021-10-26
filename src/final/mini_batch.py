"""
Mini-batch stochastic gradient descent implementation
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Logistic function
logistic = lambda z: 1./ (1 + np.exp(-z))

# Dataframe imports to synthesize accuracy reports
test = pd.read_csv("../../data/diabetes/diabetes_test.csv")
valid = pd.read_csv("../../data/diabetes/diabetes_val.csv")
# Real test and validation set features & labels
XPV = valid.drop('Outcome',axis=1).to_numpy()
XPT = test.drop('Outcome',axis=1).to_numpy()
YPV = valid['Outcome'].to_numpy().ravel()
YPT = test['Outcome'].to_numpy().ravel()

# Gradient of cost curve given inputs x,y and weights
def gradient(self,x,y,m=0,prevGrad=0):
    N, D = x.shape
    yh = logistic(np.dot(x, self.w))    # y-hat, probability class
    grad = np.dot(x.T, yh - y)/N        # divide by N because cost is mean over N points
    gradm = (m*prevGrad) + ((1-m)*grad) # calculate gradient w.r.t momentum, if m=0, then gradm=grad
    return gradm

# Batches data given a size
def BatchData(X: np.array, Y: np.array, size: int):
    assert len(X)==len(Y)
    # Iterate through data and split it based on batch size and number of batches needed
    # Function can handle iterating through the entire dataset or for certain number of batches
    for x in range(0, len(X), size):
        yield X[x : min(x + size, len(X))], Y[x : min(x + size, len(Y))]

# Logisitic regression class accounting for several input parameters for the model
class LogisticRegression:
    
    def __init__(self, add_bias=True, learning_rate=.1, epsilon=9.5e-3, max_iters=1e5, verbose=False,
                 momentum=0, batchSize=0, epochs=0, prediction=[]):
        self.add_bias = add_bias
        self.learning_rate = learning_rate
        self.epsilon = epsilon                        #to get the tolerance for the norm of gradients 
        self.max_iters = max_iters                    #maximum number of iteration of gradient descent
        self.verbose = verbose
        self.momentum = momentum
        self.batchSize = batchSize
        self.epochs = epochs
        self.prediction = prediction
    # This method fits a logisitic regression to the training data using
    # gradient descent. The weights are calculated and ouputted.
    def fit(self,x,y):
        pred=[]
        Xbatch=x
        Ybatch=y
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x,np.ones(N)])
        N,D = x.shape
        self.w = np.zeros(D)
        g = np.inf 
        t = 0
        
        if(self.batchSize==0):
            # the code snippet below is for gradient descent
            while np.linalg.norm(g) > self.epsilon and t < self.max_iters:
                if (t == 0):
                    g = self.gradient(x, y)
                else:
                    g = self.gradient(x,y,self.momentum,g)   
                self.w = self.w - self.learning_rate * g 
                t += 1
        else:
            
            # Iterate over the dataset
            for epoch in range(1, self.epochs+1):
                # Randomly shuffle the data to maximize preformance metrics
                np.random.shuffle(Xbatch)
                np.random.shuffle(Ybatch)
               
                if self.verbose:
                    print(f'Epoch: {epoch}')
                    print(f'Convergence: {t} Iterations')
                    print(f'Norm of gradient: {np.linalg.norm(g)}')
                    #print(f'\nWeights: {self.w}\n')
                    
                N,D = Xbatch.shape
                self.w = np.zeros(D)
                g = np.inf 
                t = 0
                
                # Iterating over batched data
                for X, Y in BatchData(Xbatch, Ybatch, self.batchSize):
                    # Gradient Descent loop
                    while np.linalg.norm(g) > self.epsilon and t < self.max_iters:
                        if (t == 0):
                            g = self.gradient(X, Y)
                        else:
                            g = self.gradient(X,Y,self.momentum,g)   
                        self.w = self.w - self.learning_rate * g 
                        t += 1
                        
                    # Predicitons for Test data and validation data
                    yhT = logistic(np.dot(XPT,self.w))
                    yhV = logistic(np.dot(XPV,self.w))

                    predictionT, predictionV = [],[]

                    # Decision Boundaries
                    for x in np.array(yhT).ravel():
                        if x < 0.5: predictionT.append(0)
                        else: predictionT.append(1)

                    for x in np.array(yhV).ravel():
                        if x < 0.5: predictionV.append(0)
                        else: predictionV.append(1)

                print(f'Batch Size: {self.batchSize}    TEST accuracy: {accuracy_score(YPT,predictionT)}   VALIDATION accuracy: {accuracy_score(YPV,predictionV)}')
             
        return self

# Initialize class method
LogisticRegression.gradient = gradient
