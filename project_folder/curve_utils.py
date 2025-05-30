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

