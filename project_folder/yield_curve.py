# YieldCurve, BankBill, boostrap, pricing logic 

import numpy as np
import math
import curve_utils

#Interpolate a yield curve using linear interpolation
def interpolate_yield_curve(xs, ys, x):
    xs = np.array(xs)
    ys = np.array(ys)

    # Find the interval [x0, x1] where x0 <= x <= x1
    idx = np.searchsorted(xs, x) - 1
    x0, x1 = xs[idx], xs[idx + 1]
    y0, y1 = ys[idx], ys[idx + 1]

    # Calculate the continuously compounded rate
    rate = (np.log(y1) - np.log(y0)) / (x1 - x0)

    # Calculate the interpolated yield
    y = y0 * np.exp(rate * (x - x0))
   
    return y

class ZeroCurve:
    def __init__(self):
        self.maturities = []
        self.zero_rates = []
        self.AtMats = []
        self.discount_factors = []
    
     def add_zero_rate(self, maturity, zero_rate):
        self.maturities.append(maturity)
        self.zero_rates.append(zero_rate)
        self.AtMats.append(math.exp(zero_rate * maturity))
        self.discount_factors.append(1 / self.AtMats[-1])

    def add_discount_factor(self, maturity, discount_factor):
        self.maturities.append(maturity)
        self.discount_factors.append(discount_factor)
        self.AtMats.append(1 / discount_factor)
        self.zero_rates.append(math.log(1 / discount_factor) / maturity)
    
    def get_AtMat(self, maturity):
        if maturity in self.maturities:
            return self.AtMats[self.maturities.index(maturity)]
        else:
            return exp_interp(self.maturities, self.AtMats, maturity)

    def get_discount_factor(self, maturity):
        if maturity in self.maturities:
            return self.discount_factors[self.maturities.index(maturity)]
        else:
            return exp_interp(self.maturities, self.discount_factors, maturity)

    def get_zero_rate(self, maturity):
        if maturity in self.maturities:
            return self.zero_rates[self.maturities.index(maturity)]
        else:
            return math.log(self.get_AtMat(maturity)) / maturity
        
    def get_zero_curve(self):
        return self.maturities, self.discount_factors
    
    def npv(self, cash_flows):
        npv = 0
        for maturity in cash_flows.get_maturities():
            npv += cash_flows.get_cash_flow(maturity) * self.get_discount_factor(maturity)
        return npv

# Bankbill and Portfolio classes
class BankBill:
    def __init__(self, face_value, maturity, ytm):
        self.face_value = face_value
        self.maturity = maturity
        self.ytm = ytm
        self.cash_flows = {maturity: face_value}
    
    def get_maturity(self):
        return self.maturity
    
    def get_face_value(self):
        return self.face_value
    
    def get_ytm(self):
        return self.ytm
    
    def get_cash_flows(self):
        return self.cash_flows
    
    def get_cash_flow(self, maturity):
        return self.cash_flows.get(maturity, 0)
    
    def get_price(self):
        # Price using simple discounting
        return self.face_value * math.exp(-self.ytm * self.maturity)

class Portfolio:
    def __init__(self):
        self.bank_bills = []
    
    def add_bank_bill(self, bill):
        self.bank_bills.append(bill)
    
    def get_bank_bills(self):
        return self.bank_bills
    
    def get_maturities(self):
        return [bill.get_maturity() for bill in self.bank_bills]
    
    def get_cash_flows(self):
        # Returns a dict: maturity -> total cash flow at that time
        flows = {}
        for bill in self.bank_bills:
            for mat, amt in bill.get_cash_flows().items():
                flows[mat] = flows.get(mat, 0) + amt
        return flows
    
    def get_cash_flow(self, maturity):
        return self.get_cash_flows().get(maturity, 0)
    
#Extract excel data from f17hist.xlsx
excel_path = "./project_folder/f17hist.xlsx"
curve_date = date(2025, 4, 30)

# Read the worksheet, skipping the first 10 rows (headers)
df = pd.read_excel(excel_path, sheet_name="Yields", skiprows=10)

# Rename and tidy the date column
df.rename(columns={"Series ID": "Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"]).dt.date

# Isolate the row for the desired curve date
row = df.loc[df["Date"] == curve_date]
if row.empty:
    raise ValueError("30-Apr-2025 not found – check the file or the date format!")

# Extract zero-rate columns (e.g. FZCY50D, FZCY75D, …)
zero_cols = [c for c in row.columns if re.fullmatch(r"FZCY\d+D", c)]
row_zero = row[zero_cols].iloc[0]    # turn 1-row DataFrame into a Series

# Parse tenors from column names (e.g. FZCY50D -> 50)
tenors = [int(re.findall(r"\d+", c)[0]) for c in zero_cols]  # in days
rates = row_zero.values / 100  # convert % p.a. to decimal

# Convert tenors to years for use in your curve
maturities = [t / 365.0 for t in tenors]

# --- Create BankBill objects and build the portfolio ---
bank_bills = []
for mat, rate in zip(maturities, rates):
    bill = BankBill(face_value=100, maturity=mat, ytm=rate)
    bank_bills.append(bill)

portfolio = Portfolio()
for bill in bank_bills:
    portfolio.add_bank_bill(bill)

#Bootstrap the zero curve from the portfolio
class YieldCurve(ZeroCurve):
    def __init__(self):
        super().__init__()
        self.portfolio = None

    def set_constituent_portfolio(self, portfolio):
        self.portfolio = portfolio

    def bootstrap(self):
        bank_bills = self.portfolio.get_bank_bills()
        self.add_zero_rate(0, 0)
        for bill in bank_bills:
            df = bill.get_price() / bill.get_face_value()
            self.add_discount_factor(bill.get_maturity(), df)

# --- Build and bootstrap the yield curve ---
yc = YieldCurve()
yc.set_constituent_portfolio(portfolio)
yc.bootstrap()

# --- Print the bootstrapped zero curve and risk-free rate ---
print("Bootstrapped zero curve (maturity, discount factor):")
print(yc.get_zero_curve())

# Example: Get risk-free rate from the bootstrapped curve (e.g., 1 year)
rf_1y = yc.get_zero_rate(1)
print("Risk-free zero rate at 1 year:", rf_1y)