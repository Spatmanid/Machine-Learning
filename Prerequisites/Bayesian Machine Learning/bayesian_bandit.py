import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

NUM_TRIALS = 2000
BANDIT_PROBABILITIES = [.2, .25, .3]
N = 100000

class Bandit():
    def __init__(self, p):
        self.p = p
        self.a = 1
        self.b = 1
        
    def pull(self):
        return np.random.random() < self.p
    
    def sample(self):
        return np.random.beta(self.a, self.b)
    
    def update(self, x):
        self.a += x
        self.b += 1 - x
        
def plot_bandits(bandits, trial):
    x = np.linspace(0, 1, 200)
    for b in bandits:
        y = beta.pdf(x, b.a, b.b)
        plt.plot(x, y, label="real p: %.4f" % b.p)
    plt.title("Bandit distributions after %s trials" % trial)
    plt.legend()
    plt.show()
    
def plot_cum_avg(cumulative_average_crt, p):
    plt.plot(cumulative_average_crt)
    plt.plot(np.ones(N)*p[0])
    plt.plot(np.ones(N)*p[1])
    plt.plot(np.ones(N)*p[2])
    plt.ylim((0,1))
    plt.xscale('log')
    plt.show()
    
def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999, 10000, 20000, 50000, 99999]
    data = np.empty(N)
    for i in range(N):
        all_samples = [b.sample() for b in bandits]
        j = np.argmax(all_samples)
        if i in sample_points:
            print("current samples: %s" % all_samples)
            plot_bandits(bandits, i)
        x = bandits[j].pull()
        bandits[j].update(x)
        data[i] = x
    cumulative_average_crt = np.cumsum(data) / (np.arange(N) + 1)
    plot_cum_avg(cumulative_average_crt, BANDIT_PROBABILITIES)

if __name__ == '__main__':
    experiment()
