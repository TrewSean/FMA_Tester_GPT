# derivatives.py

import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod

# ── Abstract Base ─────────────────────────────────────────────────

class Option(ABC):
    """
    Base class for all options.
    Must implement price() and delta().
    """
    def __init__(self, S0, K, T, discount):
        self.S0       = S0
        self.K        = K
        self.T        = T
        self.discount = discount

    @abstractmethod
    def price(self):
        pass

    @abstractmethod
    def delta(self):
        pass

    def vega(self, eps=1e-4):
        # Finite-difference vega
        up   = self._bump_sigma(+eps).price()
        down = self._bump_sigma(-eps).price()
        return (up - down) / (2 * eps)

    def theta(self, eps=1/365):
        # Finite-difference theta (1 day)
        orig_price = self.price()
        self.T -= eps
        price_down = self.price()
        self.T += eps
        return (price_down - orig_price) / eps

    def _bump_sigma(self, dσ):
        # Helper to clone & bump vol (works if subclass __init__ order is same)
        return self.__class__(self.S0, self.K, self.T, self.discount,
                              getattr(self, "sigma", None) + dσ, *getattr(self, "_extra_args", []))

# ── European Call ────────────────────────────────────────────────

class EuropeanCall(Option):
    def __init__(self, S0, K, T, discount, sigma):
        super().__init__(S0, K, T, discount)
        self.sigma = sigma

    def price(self):
        r = -np.log(self.discount(self.T)) / self.T
        d1 = (np.log(self.S0/self.K) + (r + 0.5*self.sigma**2)*self.T) \
             / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        df = self.discount(self.T)
        return self.S0 * norm.cdf(d1) - self.K * df * norm.cdf(d2)

    def delta(self):
        r = -np.log(self.discount(self.T)) / self.T
        d1 = (np.log(self.S0/self.K) + (r + 0.5*self.sigma**2)*self.T) \
             / (self.sigma * np.sqrt(self.T))
        return norm.cdf(d1)

    def vega(self):
        r = -np.log(self.discount(self.T)) / self.T
        d1 = (np.log(self.S0/self.K) + (r + 0.5*self.sigma**2)*self.T) \
             / (self.sigma * np.sqrt(self.T))
        return self.S0 * norm.pdf(d1) * np.sqrt(self.T)

    def theta(self):
        r = -np.log(self.discount(self.T)) / self.T
        d1 = (np.log(self.S0/self.K) + (r + 0.5*self.sigma**2)*self.T) \
             / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        df = self.discount(self.T)
        term1 = - (self.S0 * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
        term2 = r * self.K * df * norm.cdf(d2)
        return term1 - term2

# ── Barrier Base ─────────────────────────────────────────────────

class BarrierOption(Option):
    def __init__(self, S0, K, T, discount, sigma, barrier):
        super().__init__(S0, K, T, discount)
        self.sigma   = sigma
        self.barrier = barrier
        # allow bump helper to pick up sigma
        self._extra_args = (barrier,)

    @abstractmethod
    def price(self):
        pass

    def delta(self):
        eps = 1e-4 * self.S0
        up   = self.__class__(self.S0+eps, self.K, self.T,
                              self.discount, self.sigma, self.barrier).price()
        down = self.__class__(self.S0-eps, self.K, self.T,
                              self.discount, self.sigma, self.barrier).price()
        return (up - down) / (2 * eps)

# ── American Put ─────────────────────────────────────────────────

class AmericanPut(Option):
    def __init__(self, S0, K, T, discount, sigma, steps=1000):
        super().__init__(S0, K, T, discount)
        self.sigma = sigma
        self.steps = steps

    def price(self):
        dt    = self.T / self.steps
        r_eff = -np.log(self.discount(dt)) / dt
        u     = np.exp(self.sigma * np.sqrt(dt))
        d     = 1 / u
        pu    = (np.exp(r_eff*dt) - d) / (u - d)
        pd    = 1 - pu
        disc  = np.exp(-r_eff * dt)

        ST = np.array([self.S0 * u**j * d**(self.steps-j)
                       for j in range(self.steps+1)])
        payoffs = np.maximum(self.K - ST, 0)

        for i in range(self.steps-1, -1, -1):
            payoffs = disc * (pu * payoffs[1:] + pd * payoffs[:-1])
            ST = ST[:i+1] / u
            payoffs = np.maximum(payoffs, self.K - ST)
        return payoffs[0]

    def delta(self):
        eps = 1e-4 * self.S0
        up   = AmericanPut(self.S0+eps, self.K, self.T,
                           self.discount, self.sigma, self.steps).price()
        down = AmericanPut(self.S0-eps, self.K, self.T,
                           self.discount, self.sigma, self.steps).price()
        return (up - down) / (2 * eps)

# ── Up-and-In Barrier Call ────────────────────────────────────────

class UpAndInBarrierCall(BarrierOption):
    def price(self, steps=252, paths=20000):
        dt      = self.T / steps
        disc    = self.discount(self.T)
        r_eff   = -np.log(disc) / self.T
        payoffs = []
        for _ in range(paths):
            S = self.S0
            knocked = False
            for _ in range(steps):
                S *= np.exp((r_eff - 0.5*self.sigma**2)*dt
                            + self.sigma*np.sqrt(dt)*np.random.randn())
                if S >= self.barrier:
                    knocked = True
            payoffs.append(max(S - self.K, 0) if knocked else 0)
        return disc * np.mean(payoffs)

# ── Basket Call ──────────────────────────────────────────────────

class BasketCall(Option):
    def __init__(self, S0_list, weights, K, T, discount, sigma_list, corr=None):
        super().__init__(None, K, T, discount)
        self.S0_list    = np.array(S0_list)
        self.weights    = np.array(weights)
        self.sigma_list = np.array(sigma_list)
        self.corr       = corr if corr is not None else np.eye(len(S0_list))

    def price(self, steps=252, paths=20000):
        """Vectorized Monte Carlo for a European basket call."""
        dt   = self.T / steps
        L    = np.linalg.cholesky(self.corr)
        disc = self.discount(self.T)

        payoffs = []
        for _ in range(paths):
            Z = np.random.randn(len(self.S0_list), steps)
            C = L @ Z

            S = np.tile(self.S0_list[:, None], (1, steps))

            drift = ((-np.log(disc)/self.T) 
                     - 0.5 * self.sigma_list**2) * dt
            drift = drift[:, None]

            shock = self.sigma_list[:, None] * np.sqrt(dt) * C

            S = S * np.exp(drift + shock)

            ST = S[:, -1]
            payoffs.append(max(self.weights.dot(ST) - self.K, 0))

        return disc * np.mean(payoffs)

    def delta(self, eps=1e-4):
        deltas = []
        for i in range(len(self.S0_list)):
            orig = self.S0_list[i]
            up   = self.S0_list.copy(); up[i]   = orig*(1+eps)
            dn   = self.S0_list.copy(); dn[i]   = orig*(1-eps)
            price_up = BasketCall(up, self.weights, self.K,
                                  self.T, self.discount,
                                  self.sigma_list, self.corr).price()
            price_dn = BasketCall(dn, self.weights, self.K,
                                  self.T, self.discount,
                                  self.sigma_list, self.corr).price()
            deltas.append((price_up - price_dn)/(2*eps*orig))
        return np.dot(self.weights, deltas)
    
    def vega(self, eps=1e-4, steps=252, paths=20000):
        """
        Approximate basket vega by bumping all local volatilities by `eps`.
        eps: absolute bump on each sigma (e.g. 0.0001 = 1bp).
        """
        # 1) Base price
        base_price = self.price(steps=steps, paths=paths)

        # 2) Build a bumped instance with sigma_list + eps
        bumped_sigmas = np.array(self.sigma_list) + eps
        bumped = BasketCall(
            S0_list    = self.S0_list,
            weights    = self.weights,
            K          = self.K,
            T          = self.T,
            discount   = self.discount,
            sigma_list = bumped_sigmas,
            corr       = self.corr
        )

        # 3) Bumped price
        bumped_price = bumped.price(steps=steps, paths=paths)

        # 4) Finite-difference vega
        return (bumped_price - base_price) / eps
