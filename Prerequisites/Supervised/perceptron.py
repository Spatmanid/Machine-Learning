import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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

def get_data_v2():
    w = np.array([-0.5, 0.5])
    b = 0.1
    X = np.random.random((300,2)) * 2 - 1
    Y = np.sign(X.dot(w) + b)
    return X, Y

def get_xor():
    X = np.zeros((200,2))
    X[:50] = np.random.random((50,2)) / 2 + 0.5
    X[50:100] = np.random.random((50,2)) / 2
    X[100:150] = np.random.random((50,2)) / 2 + np.array([[0, 0.5]]) 
    X[150:] = np.random.random((50,2)) / 2 + np.array([[0.5, 0]])
    Y = np.array([0]*100 + [1]*100)
    return X, Y

class Perceptron:
    def fit(self, X, Y, learning_rate=1.0, epochs=1000):
        N, D = X.shape
        self.w = np.random.randn(D)
        self.b = 0
        costs = []
        for epoch in range(epochs):
            Yhat = self.predict(X)
            incorrect = np.nonzero(Y != Yhat)[0]
            if len(incorrect) == 0:
                break
            i = np.random.choice(incorrect)
            self.w += learning_rate*Y[i]*X[i]
            self.b += learning_rate*Y[i]
            c = len(incorrect) / float(N)
            costs.append(c)
        # print("final w:", self.w, "final b:", self.b, "epochs:", (epoch+1), "/", epochs)
        print("epochs:", (epoch+1), "/", epochs)
        plt.plot(costs)
        plt.show()
        
    def predict(self, X):
        return np.sign(X.dot(self.w) + self.b)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)
    
def main():
    # linearly separable data
    print("")
    print("Linearly separable data results:")
    X, Y = get_data_v2()
    plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
    plt.show()
    N_train = len(Y) // 2
    X_train, Y_train = X[:N_train], Y[:N_train]
    X_test, Y_test = X[N_train:], Y[N_train:]

    model = Perceptron()
    t0 = datetime.now()
    model.fit(X_train, Y_train)
    print("Training time:", (datetime.now() - t0))

    t0 = datetime.now()
    print("Train accuracy:", model.score(X_train, Y_train))
    print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Y_train))

    t0 = datetime.now()
    print("Test accuracy:", model.score(X_test, Y_test))
    print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Y_test))


    # mnist data
    print("")
    print("MNIST results:")
    num_data = 1000
    X, Y = get_data(num_data*2)
    X1, X2, Y1, Y2 = X[:num_data], X[num_data:], Y[:num_data], Y[num_data:]
    idx_train = np.logical_or(Y1 == 0, Y1 == 1)
    idx_test = np.logical_or(Y2 == 0, Y2 == 1)
    X_train, Y_train = X1[idx_train], Y1[idx_train]
    X_test, Y_test = X2[idx_test], Y2[idx_test]
    Y_train[Y_train == 0] = -1
    Y_test[Y_test == 0] = -1
    model = Perceptron()
    t0 = datetime.now()
    model.fit(X_train, Y_train, learning_rate=1e-2)
    print("Training time:", (datetime.now() - t0))
    t0 = datetime.now()
    print("Train accuracy:", model.score(X_train, Y_train))
    print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Y_train))
    t0 = datetime.now()
    print("Test accuracy:", model.score(X_test, Y_test))
    print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Y_test))
    
    # xor data
    print("")
    print("XOR results:")
    X, Y = get_xor()
    Y[Y == 0] = -1
    model.fit(X, Y)
    print("XOR accuracy:", model.score(X, Y))
    
if __name__ == '__main__':
    main()
