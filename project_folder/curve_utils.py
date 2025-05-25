# curve_utils.py
import numpy as np
import pandas as pd
from datetime import date
import re
from pathlib import Path

class YieldCurve:
    """Piece-wise linear zero curve (continuously-compounded)."""
    def __init__(self, pillars: np.ndarray, zero_rates: np.ndarray):
        idx = np.argsort(pillars)          # ensure ascending order
        self._pillars    = pillars[idx]
        self._zero_rates = zero_rates[idx]

    def zero_rate(self, t: float) -> float:
        return float(np.interp(t, self._pillars, self._zero_rates))

    def discount_factor(self, t: float) -> float:
        return np.exp(-self.zero_rate(t) * t)

    __call__ = discount_factor           # so you can do yc(t)

# ----------------------------------------------------------------------
def load_rba_f17_curve(excel_path: str | Path = "f17hist.xlsx",
                       use_row: date = date(2025, 4, 30)) -> YieldCurve:
    """
    Build a YieldCurve from the RBA F17 Excel download.
    Falls back to the latest earlier row if *use_row* is absent.
    """
    df = pd.read_excel(excel_path, sheet_name="Data", skiprows=10)
    df.rename(columns={"Series ID": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df.set_index("Date", inplace=True)

    zero_cols = [c for c in df.columns if re.fullmatch(r"FZCY\d+D", c)]

    if use_row in df.index:
        row = df.loc[use_row, zero_cols]
    else:
        prev = max(d for d in df.index if d < use_row)
        print(f"[curve_utils] {use_row} missing – using {prev} instead.")
        row = df.loc[prev, zero_cols]

    pillars, zeros = [], []
    for col, val in row.items():          # e.g. 'FZCY25D'
        days = int(col[4:-1])
        pillars.append(days / 365.0)
        zeros.append(val / 100.0)         # % → decimal

    return YieldCurve(np.array(pillars), np.array(zeros))
