
import numpy as np
from scipy.stats import norm

class Option:
    def __init__(self, S0, K, T, r, sigma):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

class EuropeanCall(Option):
    def price(self):
        """Black-Scholes price for European call"""
        d1 = (np.log(self.S0/self.K) + (self.r + 0.5*self.sigma**2)*self.T)/(self.sigma*np.sqrt(self.T))
        d2 = d1 - self.sigma*np.sqrt(self.T)
        return self.S0*norm.cdf(d1) - self.K*np.exp(-self.r*self.T)*norm.cdf(d2)

    def delta(self):
        d1 = (np.log(self.S0/self.K) + (self.r + 0.5*self.sigma**2)*self.T)/(self.sigma*np.sqrt(self.T))
        return norm.cdf(d1)

class AmericanPut(Option):
    def price(self, steps=1000):
        """Binomial tree price for American put"""
        dt = self.T/steps
        u = np.exp(self.sigma*np.sqrt(dt))
        d = 1/u
        pu = (np.exp(self.r*dt)-d)/(u-d)
        pd = 1-pu
        disc = np.exp(-self.r*dt)

        # initialize asset prices at maturity
        ST = np.array([self.S0 * u**j * d**(steps-j) for j in range(steps+1)])
        payoffs = np.maximum(self.K - ST, 0)

        # step backwards through tree
        for step in range(steps-1, -1, -1):
            payoffs = disc*(pu*payoffs[1:] + pd*payoffs[:-1])
            ST = ST[:step+1]/u
            payoffs = np.maximum(payoffs, self.K - ST)  # early exercise
        return payoffs[0]

class UpAndInBarrierCall(Option):
    def __init__(self, S0, K, T, r, sigma, barrier):
        super().__init__(S0, K, T, r, sigma)
        self.barrier = barrier

    def price(self, steps=252, paths=20000):
        """Monte Carlo pricing for up-and-in barrier call"""
        dt = self.T/steps
        discount = np.exp(-self.r*self.T)
        payoffs = []
        for _ in range(paths):
            S = self.S0
            knocked_in = False
            for _ in range(steps):
                S *= np.exp((self.r - 0.5*self.sigma**2)*dt + self.sigma*np.sqrt(dt)*np.random.randn())
                if S >= self.barrier:
                    knocked_in = True
            payoff = max(S - self.K, 0) if knocked_in else 0
            payoffs.append(payoff)
        return discount*np.mean(payoffs)

class BasketCall:
    def __init__(self, S0_list, weights, K, T, r, sigma_list, corr_matrix=None):
        self.S0_list = np.array(S0_list)
        self.weights = np.array(weights)
        self.K = K
        self.T = T
        self.r = r
        self.sigma_list = np.array(sigma_list)
        self.corr = corr_matrix if corr_matrix is not None else np.eye(len(S0_list))

    def price(self, steps=252, paths=20000):
        """Monte Carlo pricing for European basket call"""
        dt = self.T/steps
        L = np.linalg.cholesky(self.corr)
        discount = np.exp(-self.r*self.T)
        payoffs = []
        for _ in range(paths):
            Z = np.random.randn(len(self.S0_list), steps)
            correlated = L @ Z
            S = np.tile(self.S0_list[:, None], (1, steps))
            for t in range(steps):
                S[:, t] *= np.exp((self.r - 0.5*self.sigma_list**2)*dt + self.sigma_list*np.sqrt(dt)*correlated[:, t])
            ST = S[:, -1]
            basket = np.dot(self.weights, ST)
            payoffs.append(max(basket - self.K, 0))
        return discount*np.mean(payoffs)
