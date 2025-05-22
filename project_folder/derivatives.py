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
        # Helper to clone & bump vol
        return self.__class__(
            self.S0, self.K, self.T, self.discount,
            getattr(self, "sigma", None) + dσ,
            *getattr(self, "_extra_args", [])
        )

# ── European Call ────────────────────────────────────────────────

class EuropeanCall(Option):
    def __init__(self, S0, K, T, discount, sigma):
        super().__init__(S0, K, T, discount)
        self.sigma = sigma

    def price(self):
        r = -np.log(self.discount(self.T)) / self.T
        d1 = (np.log(self.S0/self.K) + (r + 0.5*self.sigma**2)*self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        df = self.discount(self.T)
        return self.S0 * norm.cdf(d1) - self.K * df * norm.cdf(d2)

    def delta(self):
        r = -np.log(self.discount(self.T)) / self.T
        d1 = (np.log(self.S0/self.K) + (r + 0.5*self.sigma**2)*self.T) / (self.sigma * np.sqrt(self.T))
        return norm.cdf(d1)

    def vega(self):
        r = -np.log(self.discount(self.T)) / self.T
        d1 = (np.log(self.S0/self.K) + (r + 0.5*self.sigma**2)*self.T) / (self.sigma * np.sqrt(self.T))
        return self.S0 * norm.pdf(d1) * np.sqrt(self.T)

    def theta(self):
        r = -np.log(self.discount(self.T)) / self.T
        d1 = (np.log(self.S0/self.K) + (r + 0.5*self.sigma**2)*self.T) / (self.sigma * np.sqrt(self.T))
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
        self._extra_args = (barrier,)

    @abstractmethod
    def price(self):
        pass

    def delta(self):
        eps = 1e-4 * self.S0
        up   = self.__class__(self.S0+eps, self.K, self.T, self.discount, self.sigma, self.barrier).price()
        down = self.__class__(self.S0-eps, self.K, self.T, self.discount, self.sigma, self.barrier).price()
        return (up - down) / (2 * eps)

# ── American Put (daily grid) ─────────────────────────────────────

class AmericanPut(Option):
    def __init__(self, S0, K, T, discount, sigma, steps=None):
        super().__init__(S0, K, T, discount)
        self.sigma = sigma

    def price(self):
        # daily-step binomial tree
        steps = int(round(self.T * 252))
        dt    = 1/252
        r_eff = -np.log(self.discount(dt)) / dt
        u     = np.exp(self.sigma * np.sqrt(dt))
        d     = 1 / u
        pu    = (np.exp(r_eff*dt) - d) / (u - d)
        pd    = 1 - pu
        disc  = np.exp(-r_eff * dt)

        # terminal payoffs
        ST = np.array([self.S0 * u**j * d**(steps-j) for j in range(steps+1)])
        payoffs = np.maximum(self.K - ST, 0)

        # backward induction with early exercise
        for i in range(steps-1, -1, -1):
            payoffs = disc * (pu * payoffs[1:] + pd * payoffs[:-1])
            ST      = ST[:i+1] / u
            payoffs = np.maximum(payoffs, self.K - ST)
        return payoffs[0]

    def delta(self):
        eps = 1e-4 * self.S0
        up   = AmericanPut(self.S0+eps, self.K, self.T, self.discount, self.sigma).price()
        down = AmericanPut(self.S0-eps, self.K, self.T, self.discount, self.sigma).price()
        return (up - down) / (2 * eps)

# ── Up-and-In Barrier Call (vectorized MC) ─────────────────────────

class UpAndInBarrierCall(BarrierOption):
    def price(self, paths=20000):
        """
        Vectorized Monte Carlo pricing of an up-and-in barrier call on a daily grid:
        steps = round(T*252), dt = 1/252
        """
        steps = int(round(self.T * 252))
        dt    = 1/252
        disc  = self.discount(self.T)
        r_eff = -np.log(disc) / self.T

        # 1) draw all normals at once: shape (steps, paths)
        Z = np.random.randn(steps, paths)
        # 2) compute GBM increments: shape (steps, paths)
        increments = np.exp((r_eff - 0.5*self.sigma**2)*dt + self.sigma*np.sqrt(dt)*Z)
        # 3) build price paths:       shape (steps+1, paths)
        S_paths = np.empty((steps+1, paths))
        S_paths[0] = self.S0
        S_paths[1:] = self.S0 * np.cumprod(increments, axis=0)
        # 4) barrier check per path
        knocked = (S_paths >= self.barrier).any(axis=0)
        # 5) payoffs
        payoffs = np.where(knocked, np.maximum(S_paths[-1] - self.K, 0), 0)

        return disc * payoffs.mean()

# ── Basket Call (vectorized MC) ───────────────────────────────────

class BasketCall(Option):
    def __init__(self, S0_list, weights, K, T, discount, sigma_list, corr=None):
        super().__init__(None, K, T, discount)
        self.S0_list    = np.array(S0_list)
        self.weights    = np.array(weights)
        self.sigma_list = np.array(sigma_list)
        self.corr       = corr if corr is not None else np.eye(len(S0_list))

    def price(self, paths=20000):
        """
        Vectorized Monte Carlo pricing of a European basket call on a daily grid:
        steps = round(T*252), dt = 1/252
        """
        steps = int(round(self.T * 252))
        dt    = 1/252
        disc  = self.discount(self.T)
        L     = np.linalg.cholesky(self.corr)

        # 1) draw raw normals: shape (n_assets, steps, paths)
        Z = np.random.randn(len(self.S0_list), steps, paths)
        # 2) apply correlation:     shape (n_assets, steps, paths)
        C = L @ Z

        # 3) compute drift and shock arrays
        r_eff = -np.log(disc) / self.T
        drift = ((r_eff - 0.5*self.sigma_list**2)*dt)[:, None, None]
        shock = self.sigma_list[:, None, None] * np.sqrt(dt) * C

        # 4) build simulation:      shape (n_assets, steps+1, paths)
        S0_mat = self.S0_list[:, None, None]
        factors = np.exp(drift + shock)
        S_paths = np.empty((len(self.S0_list), steps+1, paths))
        S_paths[:, 0, :] = S0_mat
        S_paths[:, 1:, :] = S0_mat * np.cumprod(factors, axis=1)

        # 5) terminal basket and payoff
        ST = S_paths[:, -1, :]                # (n_assets, paths)
        basket = self.weights.dot(ST)         # (paths,)
        payoffs = np.maximum(basket - self.K, 0)

        return disc * payoffs.mean()

    def delta(self, eps=1e-4):
        deltas = []
        for i in range(len(self.S0_list)):
            orig = self.S0_list[i]
            up_list = self.S0_list.copy(); up_list[i] = orig*(1+eps)
            dn_list = self.S0_list.copy(); dn_list[i] = orig*(1-eps)
            price_up = BasketCall(up_list, self.weights, self.K, self.T, self.discount, self.sigma_list, self.corr).price(paths=paths)
            price_dn = BasketCall(dn_list, self.weights, self.K, self.T, self.discount, self.sigma_list, self.corr).price(paths=paths)
            deltas.append((price_up - price_dn)/(2*eps*orig))
        return np.dot(self.weights, deltas)

    def vega(self, eps=1e-4, paths=20000):
        """
        Approximate basket vega by bumping all local volatilities by `eps`.
        """
        base_price = self.price(paths=paths)
        bumped_sigmas = np.array(self.sigma_list) + eps
        bumped = BasketCall(
            self.S0_list, self.weights, self.K, self.T, self.discount,
            bumped_sigmas, self.corr
        )
        bumped_price = bumped.price(paths=paths)
        return (bumped_price - base_price) / eps
