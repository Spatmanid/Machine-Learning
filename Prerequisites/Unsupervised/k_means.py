import numpy as np 
import matplotlib.pyplot as plt

def d(u, v):
    diff = u - v
    return diff.dot(diff)


def cost(X, R, M):
    cost = 0
    for k in range(len(M)):
        diff = X - M[k]
        sq_distances = (diff * diff).sum(axis=1)
        cost += (R[:,k] * sq_distances).sum()
    return cost


def plot_k_means(X, K, max_iter=20, beta=1.0, show_plots=True):
    N, D = X.shape
    M = np.zeros((K, D))
    exponents = np.empty((N, K))
    for k in range(K):
        M[k] = X[np.random.choice(N)]
    costs = np.zeros(max_iter)
    for i in range(max_iter):
        for k in range(K):
            for n in range(N):
                exponents[n,k] = np.exp(-beta*d(M[k], X[n]))
        R = exponents / exponents.sum(axis=1, keepdims=True)
        for k in range(K):
            M[k] = R[:,k].dot(X) / R[:,k].sum()
        costs[i] = cost(X, R, M)
        if i > 0:
            if np.abs(costs[i] - costs[i-1]) < 1e-5:
                break
    if show_plots:
        plt.plot(costs)
        plt.title("Costs")
        plt.show()
        random_colors = np.random.random((K, 3))
        colors = R.dot(random_colors)
        plt.scatter(X[:,0], X[:,1], c=colors)
        plt.show()
    return M, R


def get_simple_data():
    N, D, s = 900, 2, 5 
    mu1 = np.array([0, 0])
    mu2 = np.array([s, s])
    mu3 = np.array([0, s])
    X = np.zeros((N, D))
    X[:300, :] = np.random.randn(300, D) + mu1
    X[300:600, :] = np.random.randn(300, D) + mu2
    X[600:, :] = np.random.randn(300, D) + mu3
    return X


def main():
    X = get_simple_data()
    plt.scatter(X[:,0], X[:,1])
    plt.show()
    K = 3
    plot_k_means(X, K)
    K = 5 
    plot_k_means(X, K, max_iter=30)
    K = 5 
    plot_k_means(X, K, max_iter=30, beta=0.3)

    
if __name__ == '__main__':
    main()
