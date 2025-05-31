
import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod
import pandas as pd
import yfinance as yf



import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)  # Reproducibility

# Configurable Parameters
risk_free_rate = 0.05
initial_price_A = 100.0
initial_price_B = 100.0
initial_price_C = 100.0
volatility_A = 0.20
volatility_B = 0.20
volatility_C = 0.20
strike_price = 100.0
barrier_level = 80.0
maturity = 1.0
num_simulations = 10000

class YieldCurve:
    def __init__(self, rate):
        self.rate = rate
    def get_discount_factor(self, T):
        return np.exp(-self.rate * T)
yield_curve = YieldCurve(risk_free_rate)
class Share:

    def __init__(self, ticker, trade_date):
        self.ticker = ticker
        self.trade_date = trade_date
   
    def get_price(self):
        next_day = (pd.to_datetime(self.trade_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df_spot = yf.download(self.ticker, start=self.trade_date, end=next_day, progress=False)["Close"]
        price = df_spot.loc[self.trade_date]
        if isinstance(price, pd.Series):
            price = price.iloc[0]
        return float(price)

    def get_volatility(self):
        hist = yf.download(self.ticker, end=self.trade_date, period="1y", progress=False)["Close"]
        rets = hist.pct_change().dropna()
        vol = rets.std() * np.sqrt(252)
        if isinstance(vol, pd.Series):
            vol = vol.iloc[0]
        return float(vol)



# ── Abstract Option ────────────────────────────────────────────────
class Option(ABC):
    """
    Base class for all options.
    Must implement price() and delta().
    """
    def __init__(self,shareobject, K, T, discount):
        self.shareobject = shareobject
        self.S0       = shareobject.get_price()  # Current price of the underlying share
        self.K        = K
        self.T        = T
        self.discount = discount

    @abstractmethod
    def price(self):
        pass

    @abstractmethod
    def delta(self, eps=1e-4):
        pass

    def theta(self, eps=1/365):
        """
        Finite‐difference theta (per year).
        """
        orig_T     = self.T
        orig_price = self.price()

        # bump expiry down by one day
        self.T = orig_T - eps
        price_down = self.price()

        # restore original T
        self.T = orig_T

        return (price_down - orig_price) / eps


# ── Vanilla European ────────────────────────────────────────────────

class EuropeanCall(Option):
    def __init__(self, Share, K, T, discount):
        super().__init__(Share, K, T, discount)
        self.sigma = Share.get_volatility()  # Current volatility of the underlying share

    def price(self):
        r = -np.log(self.discount(self.T)) / self.T
        d1 = (np.log(self.S0/self.K) + (r + 0.5*self.sigma**2)*self.T) \
             / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        df = self.discount(self.T)
        return self.S0 * norm.cdf(d1) - self.K * df * norm.cdf(d2)

    def delta(self, eps=1e-4):
        orig_S0 = self.S0
        # bump up
        self.S0 = orig_S0 + eps
        up_price = self.price()
        # bump down
        self.S0 = orig_S0 - eps
        down_price = self.price()
        # restore
        self.S0 = orig_S0
        return (up_price - down_price) / (2 * eps)


class EuropeanPut(Option):
    def __init__(self, Share, K, T, discount):
        super().__init__(Share, K, T, discount)
        self.sigma = Share.get_volatility()

    def price(self):
        r = -np.log(self.discount(self.T)) / self.T
        d1 = (np.log(self.S0/self.K) + (r + 0.5*self.sigma**2)*self.T) \
             / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        df = self.discount(self.T)
        return self.K * df * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)

    def delta(self, eps=1e-4):
        orig_S0 = self.S0
        self.S0 = orig_S0 + eps
        up_price = self.price()
        self.S0 = orig_S0 - eps
        down_price = self.price()
        self.S0 = orig_S0
        return (up_price - down_price) / (2 * eps)


# ── American Put via CRR ─────────────────────────────────────────────

class AmericanPut(Option):
    def __init__(self, Share, K, T, discount):
        super().__init__(Share, K, T, discount)
        self.sigma = Share.get_volatility()  # Current volatility of the underlying share

    def price(self):
        """
        Price an American‐style put via a CRR binomial tree on a daily grid.
        """
        # steps ≈ trading days
        steps = int(round(self.T * 252))
        dt    = self.T / steps

        # implied continuous rate
        r = -np.log(self.discount(self.T)) / self.T

        # up/down factors
        u  = np.exp(self.sigma * np.sqrt(dt))
        d  = 1 / u

        # risk‐neutral probabilities
        pu = (np.exp(r * dt) - d) / (u - d)
        pd = 1 - pu
        df = np.exp(-r * dt)

        # stock price tree at maturity
        j = np.arange(steps + 1)
        S = self.S0 * (u**j) * (d**(steps - j))

        # option value at maturity
        V = np.maximum(self.K - S, 0)

        # backward induction
        for i in range(steps - 1, -1, -1):
            # roll back continuation value
            V = df * (pu * V[1:] + pd * V[:-1])

            # underlying tree at time i
            S = self.S0 * (u**np.arange(i + 1)) * (d**(i + 1 - np.arange(i + 1)))

            # check early exercise
            V = np.maximum(V, self.K - S)

        return V[0]

    def delta(self, eps=1e-4):
        orig_S0 = self.S0
        self.S0 = orig_S0 + eps
        up_price = self.price()
        self.S0 = orig_S0 - eps
        down_price = self.price()
        self.S0 = orig_S0
        return (up_price - down_price) / (2 * eps)


# ── Single‐Barrier (Up‐and‐In) ────────────────────────────────────────

class BarrierOption(Option):
    def __init__(self, Share, K, T, discount, barrier):
        super().__init__(Share, K, T, discount)
        self.sigma  = Share.get_volatility()  # Current volatility of the underlying share
        self.barrier = barrier

    def price(self):
        """
        Black‐Scholes price for a European up‐and‐in call (analytic).
        """
        r  = -np.log(self.discount(self.T)) / self.T
        mu = (r - 0.5*self.sigma**2) / (self.sigma**2)
        lambda_ = np.sqrt(mu**2 + 2*r/self.sigma**2)
        x1 = (np.log(self.S0/self.barrier) / (self.sigma*np.sqrt(self.T))
              + lambda_*self.sigma*np.sqrt(self.T))
        y = (np.log(self.barrier**2/(self.S0*self.K)) \
             / (self.sigma*np.sqrt(self.T))
             + lambda_*self.sigma*np.sqrt(self.T))
        A = (self.S0 * (self.barrier/self.S0)**(2*mu) *
             norm.cdf(-y) - self.K * np.exp(-r*self.T) *
             (self.barrier/self.S0)**(2*mu-2) * norm.cdf(-y + self.sigma*np.sqrt(self.T)))
        B = (self.S0 * norm.cdf(x1) - self.K * np.exp(-r*self.T) *
             norm.cdf(x1 - self.sigma*np.sqrt(self.T)))
        return B - A

    def delta(self, eps=1e-4):
        """
        Finite‐difference delta for barrier options (central difference).
        """
        orig_S0 = self.S0

        # bump up
        self.S0 = orig_S0 + eps
        up_price = self.price()

        # bump down
        self.S0 = orig_S0 - eps
        down_price = self.price()

        # restore
        self.S0 = orig_S0

        return (up_price - down_price) / (2 * eps)


# ── Basket Call via Monte Carlo ──────────────────────────────────────

class BasketCall(Option):
    def __init__(self, share_list, weights, K, T, discount, corr, paths=100000):
        self.share_list = share_list
        self.S0_list    = np.array([s.get_price() for s in share_list])
        self.sigma_list = np.array([s.get_volatility() for s in share_list])
        self.weights    = np.array(weights)
        self.K          = K
        self.T          = T
        self.discount   = discount
        self.corr       = np.array(corr)
        self.paths      = paths
    # (rest of your BasketCall methods unchanged)


    def price(self, paths=None):
        if paths is None:
            paths = self.paths

        # Cholesky factor
        L = np.linalg.cholesky(self.corr)

        # simulate correlated normals
        Z = np.random.standard_normal((paths, len(self.S0_list)))
        correlated = Z.dot(L.T)

        # drift & diffusion
        r = -np.log(self.discount(self.T)) / self.T
        drift = (r - 0.5*self.sigma_list**2)*self.T
        diffusion = self.sigma_list * np.sqrt(self.T) * correlated

        # simulate terminal stock prices
        ST = self.S0_list * np.exp(drift + diffusion)

        payoff = np.maximum(np.dot(ST, self.weights) - self.K, 0)
        return self.discount(self.T) * np.mean(payoff)


    def delta(self, eps=1e-4, paths=None):
        """
        Returns the array of partial deltas for each underlying asset.
        """
        base_price = self.price(paths=paths)
        n = len(self.S0_list)
        partials = np.zeros(n)

        for i in range(n):
            class BumpedShare:
                def __init__(self, price, vol):
                    self._price = price
                    self._vol = vol
                def get_price(self):
                    return self._price
                def get_volatility(self):
                    return self._vol

            bumped_shares = [
                BumpedShare(self.S0_list[j] + (eps if i == j else 0), self.sigma_list[j])
                for j in range(n)
            ]

            bumped = BasketCall(
                share_list = bumped_shares,
                weights    = self.weights,
                K          = self.K,
                T          = self.T,
                discount   = self.discount,
                corr       = self.corr,
                paths      = paths or self.paths
            )
            partials[i] = (bumped.price(paths=paths) - base_price) / eps

        return partials  # <-- now returns the vector!


    def vega(self, eps=1e-4, paths=None):
        """
        Approximate basket vega by bumping all local volatilities by `eps`.
        """
        base_price = self.price(paths=paths)
        n = len(self.sigma_list)
        # Build a bumped share_list with increased volatility
        class BumpedShare:
            def __init__(self, price, vol):
                self._price = price
                self._vol = vol
            def get_price(self):
                return self._price
            def get_volatility(self):
                return self._vol

        bumped_shares = [
            BumpedShare(self.S0_list[j], self.sigma_list[j] + eps)
            for j in range(n)
        ]

        bumped = BasketCall(
            share_list = bumped_shares,
            weights    = self.weights,
            K          = self.K,
            T          = self.T,
            discount   = self.discount,
            corr       = self.corr,
            paths      = paths or self.paths
        )
        bumped_price = bumped.price(paths=paths)
        return (bumped_price - base_price) / eps