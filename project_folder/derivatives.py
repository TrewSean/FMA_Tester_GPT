import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod
from pandas import pd
from yfinance import yf

# ── Abstract Base ─────────────────────────────────────────────────

###create class for share
class Share:
    def __init__(self, ticker, trade_date):
        self.ticker = ticker
        self.trade_date = trade_date

    def get_price(self):
        
        next_day   = (pd.to_datetime(self.trade_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df_spot  = yf.download(self.ticker, start=self.trade_date, end=next_day, progress=False)["Close"]
        S0       = df_spot.loc[self.trade_date].to_dict()  # spot prices dict
        return S0

    def get_volatility(self):
        hist = yf.download(self.ticker, end=self.trade_date, period="1y", progress=False)["Close"]
        rets = hist.pct_change().dropna()                # daily returns
        vol  = (rets.std() * np.sqrt(252)).to_dict()     # annualised vol σ√252
    
        return vol


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
    def __init__(self, S0, K, T, discount, sigma):
        super().__init__(S0, K, T, discount)
        self.sigma = sigma

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
    def __init__(self, S0, K, T, discount, sigma):
        super().__init__(S0, K, T, discount)
        self.sigma = sigma

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
    def __init__(self, S0, K, T, discount, sigma, barrier):
        super().__init__(S0, K, T, discount)
        self.sigma  = sigma
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
    def __init__(self, S0_list, weights, K, T, discount,
                 sigma_list, corr, paths=100000):
        """
        Monte‐Carlo basket call.
        corr should be positive‐definite.
        """
        # treat lists as vectors
        self.S0_list    = np.array(S0_list)
        self.weights    = np.array(weights)
        self.K          = K
        self.T          = T
        self.discount   = discount
        self.sigma_list = np.array(sigma_list)
        self.corr       = np.array(corr)
        self.paths      = paths

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
        Portfolio delta = sum_i w_i * ∂P/∂S_i via finite differences.
        """
        base_price = self.price(paths=paths)
        n          = len(self.S0_list)
        partials   = np.zeros(n)

        # bump each asset in turn
        for i in range(n):
            bumped_S0 = self.S0_list.copy()
            bumped_S0[i] += eps

            bumped = BasketCall(
                S0_list    = bumped_S0,
                weights    = self.weights,
                K          = self.K,
                T          = self.T,
                discount   = self.discount,
                sigma_list = self.sigma_list,
                corr       = self.corr,
                paths      = paths or self.paths
            )
            partials[i] = (bumped.price(paths=paths) - base_price) / eps

        # portfolio (scalar) delta
        portfolio_delta = np.dot(self.weights, partials)
        return portfolio_delta


    def vega(self, eps=1e-4, paths=None):
        """
        Approximate basket vega by bumping all local volatilities by `eps`.
        """
        base_price = self.price(paths=paths)
        bumped_sigmas = self.sigma_list + eps
        bumped = BasketCall(
            S0_list    = self.S0_list,
            weights    = self.weights,
            K          = self.K,
            T          = self.T,
            discount   = self.discount,
            sigma_list = bumped_sigmas,
            corr       = self.corr,
            paths      = paths or self.paths
        )
        bumped_price = bumped.price(paths=paths)
        return (bumped_price - base_price) / eps


import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)  # Reproducibility

# Configurable Parameters
risk_free_rate = 0.05
initial_price_A = 100.0
initial_price_B = 100.0
volatility_A = 0.20
volatility_B = 0.20
strike_price = 100.0
barrier_level = 80.0
maturity = 1.0
num_simulations = 10000
steps_for_barrier = 100

class YieldCurve:
    def __init__(self, rate):
        self.rate = rate
    def get_discount_factor(self, T):
        return np.exp(-self.rate * T)

yield_curve = YieldCurve(risk_free_rate)


class Share:
    def __init__(self, name, initial_price, volatility):
        self.name = name
        self.initial_price = initial_price
        self.volatility = volatility

    def simulate_price(self, T, n_paths=1):
        r = risk_free_rate
        sigma = self.volatility
        Z = np.random.normal(size=n_paths)
        return self.initial_price * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    def simulate_path(self, T, n_steps, n_paths=1):
        r = risk_free_rate
        sigma = self.volatility
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.initial_price
        for step in range(1, n_steps + 1):
            Z = np.random.normal(size=n_paths)
            paths[:, step] = paths[:, step-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        return paths

class ShareOption:
    def __init__(self, underlying_share, strike, maturity, option_type="call", yield_curve=yield_curve):
        self.underlying = underlying_share
        self.strike = strike
        self.maturity = maturity
        self.option_type = option_type.lower()
        self.yield_curve = yield_curve

    def price_option(self, num_simulations=10000):
        final_prices = self.underlying.simulate_price(self.maturity, n_paths=num_simulations)
        if self.option_type == "call":
            payoffs = np.maximum(final_prices - self.strike, 0.0)
        elif self.option_type == "put":
            payoffs = np.maximum(self.strike - final_prices, 0.0)
        else:
            raise ValueError("Invalid option type.")
        return self.yield_curve.get_discount_factor(self.maturity) * np.mean(payoffs)

class BarrierShareOption(ShareOption):
    def __init__(self, underlying_share, strike, maturity, barrier_level, barrier_type="down_out", option_type="call", yield_curve=yield_curve):
        super().__init__(underlying_share, strike, maturity, option_type, yield_curve)
        self.barrier_level = barrier_level
        self.barrier_type = barrier_type

    def price_option(self, num_simulations=10000, num_steps=100):
        paths = self.underlying.simulate_path(self.maturity, n_steps=num_steps, n_paths=num_simulations)
        final_prices = paths[:, -1]
        if self.barrier_type == "down_out":
            knocked_mask = (np.min(paths, axis=1) <= self.barrier_level)
        elif self.barrier_type == "up_out":
            knocked_mask = (np.max(paths, axis=1) >= self.barrier_level)
        else:
            raise ValueError("Invalid barrier type.")
        if self.option_type == "call":
            payoffs = np.maximum(final_prices - self.strike, 0.0)
        else:
            payoffs = np.maximum(self.strike - final_prices, 0.0)
        payoffs[knocked_mask] = 0.0
        return self.yield_curve.get_discount_factor(self.maturity) * np.mean(payoffs)

class BasketShareOption(ShareOption):
    def __init__(self, underlying_shares, weights, strike, maturity, option_type="call", yield_curve=yield_curve):
        super().__init__(underlying_shares[0], strike, maturity, option_type, yield_curve)
        self.underlying_shares = underlying_shares
        self.weights = weights

    def price_option(self, num_simulations=10000):
        n = num_simulations
        m = len(self.underlying_shares)
        final_prices_matrix = np.zeros((n, m))
        for j, share in enumerate(self.underlying_shares):
            final_prices_matrix[:, j] = share.simulate_price(self.maturity, n_paths=n)
        basket_values = final_prices_matrix.dot(np.array(self.weights))
        if self.option_type == "call":
            payoffs = np.maximum(basket_values - self.strike, 0.0)
        else:
            payoffs = np.maximum(self.strike - basket_values, 0.0)
        return self.yield_curve.get_discount_factor(self.maturity) * np.mean(payoffs)


# Create shares
stock_A = Share("Stock A", initial_price=initial_price_A, volatility=volatility_A)
stock_B = Share("Stock B", initial_price=initial_price_B, volatility=volatility_B)

# Plain European call option
plain_call = ShareOption(stock_A, strike_price, maturity, "call")
plain_price = plain_call.price_option(num_simulations)
print(f"Plain Call Option Price: {plain_price:.4f}")

# Barrier option
barrier_call = BarrierShareOption(stock_A, strike_price, maturity, barrier_level, "down_out", "call")
barrier_price = barrier_call.price_option(num_simulations, steps_for_barrier)
print(f"Barrier Call Option Price: {barrier_price:.4f}")

# Basket option
basket_call = BasketShareOption([stock_A, stock_B], [0.5, 0.5], strike_price, maturity, "call")
basket_price = basket_call.price_option(num_simulations)
print(f"Basket Call Option Price: {basket_price:.4f}")



import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. SETUP: Financial and Simulation Parameters
# ===============================

np.random.seed(42)  # Fix random seed so results are repeatable

# Market conditions
risk_free_rate = 0.05        # "Safe" interest rate used for discounting (e.g., from a bank)
maturity = 1.0               # Option lifespan in years (e.g., 1 year until expiration)

# Stock settings
initial_price_A = 100.0      # Starting price of Stock A
initial_price_B = 100.0      # Starting price of Stock B
volatility_A = 0.2           # Annual volatility (standard deviation of returns) of Stock A
volatility_B = 0.2           # Volatility of Stock B

# Option contract details
strike_price = 100.0         # Exercise price for all options
barrier_level = 90        # If Stock A drops below this, the barrier option is knocked out

# Simulation settings
num_steps = 252              # Number of time intervals (252 trading days in a year)
num_simulations = 10000      # Number of simulated price paths
dt = maturity / num_steps    # Time step size (1 day in years)

# ===============================
# 2. SIMULATE STOCK PRICE PATHS
# ===============================

# Initialize empty arrays for stock prices
S_A = np.zeros((num_simulations, num_steps + 1))  # Rows = paths, Columns = time steps
S_B = np.zeros((num_simulations, num_steps + 1))

# Set starting price for all paths at time zero
S_A[:, 0] = initial_price_A
S_B[:, 0] = initial_price_B

# Simulate price paths using Geometric Brownian Motion
for t in range(1, num_steps + 1):
    Z_A = np.random.normal(size=num_simulations)  # Random shocks for Stock A
    Z_B = np.random.normal(size=num_simulations)  # Random shocks for Stock B

    # Update price for each path at time t
    S_A[:, t] = S_A[:, t - 1] * np.exp((risk_free_rate - 0.5 * volatility_A**2) * dt + volatility_A * np.sqrt(dt) * Z_A)
    S_B[:, t] = S_B[:, t - 1] * np.exp((risk_free_rate - 0.5 * volatility_B**2) * dt + volatility_B * np.sqrt(dt) * Z_B)

# ===============================
# 3. CALCULATE OPTION PAYOFFS
# ===============================

# Final prices at maturity (last column of simulated paths)
final_prices_A = S_A[:, -1]
final_prices_B = S_B[:, -1]

# --- 1. Plain European Call: max(S_T - K, 0) ---
plain_call_payoff = np.maximum(final_prices_A - strike_price, 0)

# --- 2. Down-and-Out Barrier Call ---
# If the stock price ever goes below the barrier during the year, the option dies (pays 0)
knocked_out = np.any(S_A <= barrier_level, axis=1)  # Check each path: did it touch or go below the barrier?
barrier_call_payoff = np.where(knocked_out, 0, np.maximum(final_prices_A - strike_price, 0))

# --- 3. Basket Call: Based on average of Stock A and B at maturity ---
basket_final = 0.5 * final_prices_A + 0.5 * final_prices_B
basket_call_payoff = np.maximum(basket_final - strike_price, 0)

# ===============================
# 4. SORT FOR SMOOTH PLOTTING
# ===============================

# Sort paths by final Stock A price (x-axis) so curves appear smooth
sorted_idx = np.argsort(final_prices_A)
sorted_prices = final_prices_A[sorted_idx]  # x-axis

# Match the sorted order for each payoff
plain_sorted = plain_call_payoff[sorted_idx]
barrier_sorted = barrier_call_payoff[sorted_idx]
basket_sorted = basket_call_payoff[sorted_idx]

# ===============================
# 5. PLOT: INDIVIDUAL PAYOFFS
# ===============================

plt.figure(figsize=(12, 12))  # Bigger figure for readability

# --- 1. Plain Call ---
plt.subplot(3, 1, 1)
plt.plot(sorted_prices, plain_sorted, color='blue', linewidth=2, label='Plain Call')
plt.axvline(strike_price, color='gray', linestyle='--', label='Strike Price')
plt.axhline(0, color='black', linewidth=0.8)
plt.title("Plain European Call Option")
plt.xlabel("Final Stock A Price")
plt.ylabel("Payoff")
plt.grid(True)
plt.legend()

# --- 2. Barrier Call ---
plt.subplot(3, 1, 2)
plt.plot(sorted_prices, barrier_sorted, color='orange', linewidth=2, label='Barrier Call (Down-and-Out)')
plt.axvline(strike_price, color='gray', linestyle='--', label='Strike Price')
plt.axvline(barrier_level, color='red', linestyle='--', label='Barrier Level')
plt.axhline(0, color='black', linewidth=0.8)
plt.title("Down-and-Out Barrier Call Option")
plt.xlabel("Final Stock A Price")
plt.ylabel("Payoff")
plt.grid(True)
plt.legend()

# --- 3. Basket Call ---
plt.subplot(3, 1, 3)
plt.plot(sorted_prices, basket_sorted, color='green', linewidth=0.8, label='Basket Call')
plt.axvline(strike_price, color='gray', linestyle='--', label='Strike Price')
plt.axhline(0, color='black', linewidth=0.8)
plt.title("Basket Call Option")
plt.xlabel("Final Stock A Price")
plt.ylabel("Payoff")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ===============================
# 6. PLOT: COMBINED PAYOFFS
# ===============================

plt.figure(figsize=(10, 6))  # Smaller single plot

plt.plot(sorted_prices, plain_sorted, label="Plain Call", linewidth=2)
plt.plot(sorted_prices, barrier_sorted, label="Barrier Call (Down-and-Out)", linewidth=2)
plt.plot(sorted_prices, basket_sorted, label="Basket Call", linewidth=0.8)

# Reference lines
plt.axvline(strike_price, color='gray', linestyle='--', label='Strike Price')
plt.axvline(barrier_level, color='red', linestyle='--', label='Barrier Level')
plt.axhline(0, color='black', linewidth=0.8)

# Titles and labels
plt.title("Comparison of Option Payoffs")
plt.xlabel("Final Stock A Price")
plt.ylabel("Payoff")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




