

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
stock_C = Share("Stock C", initial_price=initial_price_C, volatility=volatility_C)
# Plain European call option
plain_call = ShareOption(stock_A, strike_price, maturity, "call")
plain_price = plain_call.price_option(num_simulations)
print(f"Plain Call Option Price: {plain_price:.4f}")
# Barrier option
barrier_call = BarrierShareOption(stock_A, strike_price, maturity, barrier_level, "down_out", "call")
barrier_price = barrier_call.price_option(num_simulations)
print(f"Barrier Call Option Price: {barrier_price:.4f}")
# Basket option
basket_call = BasketShareOption([stock_A, stock_B, stock_C], [0.4, 0.3, 0.3], strike_price, maturity, "call")
basket_price = basket_call.price_option(num_simulations)
print(f"Basket Call Option Price: {basket_price:.4f}")



import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)  # Reproducibility

# Create Diagrams
# ── Parameters ─────────────────────────────────────────────────────

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
initial_price_C = 100.0      # Starting price of Stock C
volatility_A = 0.2           # Annual volatility (standard deviation of returns) of Stock A
volatility_B = 0.2           # Volatility of Stock B
volatility_C = 0.2           # Volatility of Stock C

# Option contract details
strike_price = 100.0         # Exercise price for all options
barrier_level = 90.0         # If Stock A drops below this, the barrier option is knocked out

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
S_C = np.zeros((num_simulations, num_steps + 1))

# Set starting price for all paths at time zero
S_A[:, 0] = initial_price_A
S_B[:, 0] = initial_price_B
S_C[:, 0] = initial_price_C

# Simulate price paths using Geometric Brownian Motion
for t in range(1, num_steps + 1):
    Z_A = np.random.normal(size=num_simulations)  # Random shocks for Stock A
    Z_B = np.random.normal(size=num_simulations)  # Random shocks for Stock B
    Z_C = np.random.normal(size=num_simulations)  # Random shocks for Stock C

    # Update price for each path at time t
    S_A[:, t] = S_A[:, t - 1] * np.exp((risk_free_rate - 0.5 * volatility_A**2) * dt + volatility_A * np.sqrt(dt) * Z_A)
    S_B[:, t] = S_B[:, t - 1] * np.exp((risk_free_rate - 0.5 * volatility_B**2) * dt + volatility_B * np.sqrt(dt) * Z_B)
    S_C[:, t] = S_C[:, t - 1] * np.exp((risk_free_rate - 0.5 * volatility_C**2) * dt + volatility_C * np.sqrt(dt) * Z_C)

# ===============================
# 3. CALCULATE OPTION PAYOFFS
# ===============================

# Final prices at maturity (last column of simulated paths)
final_prices_A = S_A[:, -1]
final_prices_B = S_B[:, -1]
final_prices_C = S_C[:, -1]

# --- 1. Plain European Call: max(S_T - K, 0) ---
plain_call_payoff = np.maximum(final_prices_A - strike_price, 0)

# --- 2. Down-and-Out Barrier Call ---
# If the stock price ever goes below the barrier during the year, the option dies (pays 0)
knocked_out = np.any(S_A <= barrier_level, axis=1)  # Check each path: did it touch or go below the barrier?
barrier_call_payoff = np.where(knocked_out, 0, np.maximum(final_prices_A - strike_price, 0))

# --- 3. Basket Call: Based on average of Stock A, B, and C at maturity ---
basket_final = (1/3) * (final_prices_A + final_prices_B + final_prices_C)
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
plt.plot(sorted_prices, basket_sorted, color='green', linewidth=2, label='Basket Call')
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
plt.plot(sorted_prices, basket_sorted, label="Basket Call", linewidth=2)

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
