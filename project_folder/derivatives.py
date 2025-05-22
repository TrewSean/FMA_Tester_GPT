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

# ── American Put (daily grid) ─────────────────────────────────────
class AmericanPut(Option):
    def __init__(self, S0, K, T, discount, sigma, steps=None):
        super().__init__(S0, K, T, discount)
        self.sigma = sigma

    def price(self):
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
        steps = int(round(self.T * 252))
        dt    = 1/252
        disc  = self.discount(self.T)
        r_eff = -np.log(disc) / self.T

        # draw normals & simulate
        Z = np.random.randn(steps, paths)
        increments = np.exp((r_eff - 0.5*self.sigma**2)*dt + self.sigma*np.sqrt(dt)*Z)
        S_paths = np.empty((steps+1, paths))
        S_paths[0] = self.S0
        S_paths[1:] = self.S0 * np.cumprod(increments, axis=0)

        # payoff
        knocked = (S_paths >= self.barrier).any(axis=0)
        payoffs = np.where(knocked, np.maximum(S_paths[-1] - self.K, 0), 0)
        return disc * payoffs.mean()

    def delta(self, paths=20000):
        # pathwise (score-function) delta estimator
        steps = int(round(self.T * 252)); dt = 1/252
        disc  = self.discount(self.T); r_eff = -np.log(disc) / self.T

        Z = np.random.randn(steps, paths)
        increments = np.exp((r_eff - 0.5*self.sigma**2)*dt + self.sigma*np.sqrt(dt)*Z)
        S_paths = np.empty((steps+1, paths))
        S_paths[0] = self.S0
        S_paths[1:] = self.S0 * np.cumprod(increments, axis=0)

        ST      = S_paths[-1]
        knocked = (S_paths >= self.barrier).any(axis=0)
        itm     = ST > self.K
        delta_samps = (ST / self.S0) * (knocked & itm)
        return disc * delta_samps.mean()

# ── Basket Call (vectorized MC) ───────────────────────────────────
class BasketCall(Option):
    def __init__(self, S0_list, weights, K, T, discount, sigma_list, corr=None):
        super().__init__(None, K, T, discount)
        self.S0_list    = np.array(S0_list)
        self.weights    = np.array(weights)
        self.sigma_list = np.array(sigma_list)
        self.corr       = corr if corr is not None else np.eye(len(S0_list))

    def price(self, paths=20000):
        steps = int(round(self.T * 252)), dt = 1/252
        disc  = self.discount(self.T)
        L     = np.linalg.cholesky(self.corr)

        n = len(self.S0_list)
        Z = np.random.randn(n, steps, paths)
        C = np.tensordot(L, Z, axes=[1, 0])

        r_eff = -np.log(disc) / self.T
        drift = ((r_eff - 0.5*self.sigma_list**2)*dt)[:, None, None]
        shock = self.sigma_list[:, None, None] * np.sqrt(dt) * C

        factors = np.exp(drift + shock)
        S_paths = np.zeros((n, steps+1, paths))
        S_paths[:, 0, :] = self.S0_list[:, None]
        S_paths[:, 1:, :] = S_paths[:, 0:1, :] * np.cumprod(factors, axis=1)

        ST = S_paths[:, -1, :]
        basket = self.weights.dot(ST)
        payoffs = np.maximum(basket - self.K, 0)
        return disc * payoffs.mean()

    def delta(self, paths=20000):
        # pathwise delta for basket
        steps = int(round(self.T * 252)); dt = 1/252
        disc  = self.discount(self.T)
        L     = np.linalg.cholesky(self.corr)

        n = len(self.S0_list)
        Z = np.random.randn(n, steps, paths)
        C = np.tensordot(L, Z, axes=[1, 0])
        r_eff = -np.log(disc) / self.T
        drift = ((r_eff - 0.5*self.sigma_list**2)*dt)[:, None, None]
        shock = self.sigma_list[:, None, None] * np.sqrt(dt) * C
        factors = np.exp(drift + shock)

        S_paths = np.zeros((n, steps+1, paths))
        S_paths[:, 0, :] = self.S0_list[:, None]
        S_paths[:, 1:, :] = S_paths[:, 0:1, :] * np.cumprod(factors, axis=1)

        ST = S_paths[:, -1, :]
        basket = self.weights.dot(ST)
        payoff_mask = basket > self.K

        # pathwise delta_i = (ST_i / S0_i) * 1{payoff>0}
        delta_samples = (ST / self.S0_list[:, None]) * payoff_mask
        # weighted sum across assets
        weighted = self.weights[:, None] * delta_samples
        return disc * weighted.sum(axis=0).mean()

    def vega(self, eps=1e-4, paths=20000):
        base_price = self.price(paths=paths)
        bumped_sigmas = np.array(self.sigma_list) + eps
        bumped = BasketCall(self.S0_list, self.weights, self.K, self.T, self.discount, bumped_sigmas, self.corr)
        bumped_price = bumped.price(paths=paths)
        return (bumped_price - base_price) / eps
