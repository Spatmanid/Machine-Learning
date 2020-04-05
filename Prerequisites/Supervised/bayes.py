import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sortedcontainers import SortedList
from datetime import datetime
from scipy.stats import multivariate_normal as mvn


def get_data(limit=None):
    print("Loading data...")
    df = pd.read_csv('train.csv')
    data = df.values
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y
    
class Bayes:
    def __init__(self):
        self.gaussians = {}
        self.priors = {}
    
    def fit(self, X, Y, smoothing=10e-3):
        N, D = X.shape
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis=0),
                'cov': np.cov(current_x.T) + np.eye(D) * smoothing,
            }
            self.priors[c] = float(len(Y[Y == c])) / len(Y)
            
    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in self.gaussians.items():
            mean, cov = g['mean'], g['cov']
            P[:,c] = mvn.logpdf(X, mean=mean, cov=cov) + np.log(self.priors[c])      
        return np.argmax(P, axis=1)
    
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

def main():   
    X, Y = get_data(20000)
    Ntrain = len(Y) // 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = Bayes()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print("Training time:", (datetime.now() - t0))

    t0 = datetime.now()
    print("Train accuracy:", model.score(Xtrain, Ytrain))
    print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

    t0 = datetime.now()
    print("Test accuracy:", model.score(Xtest, Ytest))
    print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))

if __name__ == "__main__":
    main()
