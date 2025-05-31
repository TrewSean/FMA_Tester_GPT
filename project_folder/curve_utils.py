import pandas as pd
from datetime import date
import re

# 1)  file name
excel_path = "./project_folder/f17hist.xlsx"      # <-- points to ./data/f17hist.xlsx
curve_date = date(2025, 4, 30)

# 2)  read the worksheet
df = pd.read_excel(excel_path,
                   sheet_name="Yields",   # <- this sheet holds the zero-rates
                   skiprows=10)           # RBA puts headers in the first 10 rows

print(df.columns)
print(df.head())

# 3)  tidy the date column
df.rename(columns={"Series ID": "Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"]).dt.date

# 4)  isolate the 30-Apr row
row = df.loc[df["Date"] == curve_date]
print(row)

if row.empty:
    raise ValueError("30-Apr-2025 not found – check the file or the date format!")

# 5)  keep only zero-rate columns (e.g. FZCY50D, FZCY75D, …)
zero_cols = [c for c in row.columns if re.fullmatch(r"FZCY\d+D", c)]
row_zero  = row[zero_cols].iloc[0]    # turn 1-row DataFrame into a Series

print("First few zero-rates (% p.a.):")
print(row_zero.head())


# next steps:

# ...existing code...

import numpy as np

# 6) Parse maturities (in years) and rates (as decimals)
tenors = [int(re.findall(r"\d+", c)[0]) for c in zero_cols]  # e.g. 50, 75, ...
maturities = [t / 365.0 for t in tenors]  # convert days to years
rates = row_zero.values / 100  # convert % p.a. to decimal

def interpolate_zero_rate(maturity):
    """Linearly interpolate zero rate for a given maturity (in years)."""
    xs = np.array(maturities)
    ys = np.array(rates)
    if maturity <= xs[0]:
        return ys[0]
    if maturity >= xs[-1]:
        return ys[-1]
    idx = np.searchsorted(xs, maturity) - 1
    x0, x1 = xs[idx], xs[idx + 1]
    y0, y1 = ys[idx], ys[idx + 1]
    # Linear interpolation
    return y0 + (y1 - y0) * (maturity - x0) / (x1 - x0)

# Example: print the interpolated zero rate for 2 years
print("Zero rate for 2 years:", interpolate_zero_rate(2))


# discount factors

def discount_factor(T):
    """Return discount factor for maturity T (in years) using interpolated zero rate."""
    r = interpolate_zero_rate(T)
    return np.exp(-r * T)

# Example: print the discount factor for 2 years
print("Discount factor for 2 years:", discount_factor(2))

# ...existing code...

def get_discount_function():
    """
    Returns a discount function discount(T) using the loaded/interpolated zero curve.
    """
    def discount(T):
        r = interpolate_zero_rate(T)
        return np.exp(-r * T)
    return discount

