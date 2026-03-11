# Import packages
import argparse
import yaml
from pathlib import Path
import time
from src import BESS_trading_strategy, API_KEY_ENTSOE
from src.utils import previous_day_str
from datetime import date
from entsoe import EntsoePandasClient
import numpy as np
import pandas as pd

# Initialize the request protocol
CLIENT = EntsoePandasClient(api_key=API_KEY_ENTSOE)

# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("VT_name", type=str)
args = parser.parse_args()
# Store the parsed name of the VT
VT_name = args.VT_name


## Virtual Trader work

# Read config
with open(f"VT/VT_config/VTSE_{VT_name}.yml", "r") as f:
    config = yaml.safe_load(f)

# Date of Trade - Trading for previous day for now since forecasting module is not yet ready
date_trade = previous_day_str(date.today().strftime("%Y-%m-%d"))

# Runtime for data loading
start_data = time.time()

# Create trading instance
trade = BESS_trading_strategy(
            CLIENT=CLIENT,
            SOC_max=config["SOC_max"],
            R_max=config["R_max"],
            SOC_i=config["SOC_i"],
            FCR_max=config["FCR_max"],
            scheme=config["scheme"],
            W=config["W"],
            markets=config["markets"],
            country=config["country"],
            start_date=date_trade,
            end_date=date_trade,
        )

# Runtime for data loading
runtime_data = time.time() - start_data

# Runtime for optimization
start_opt = time.time()

# Solve the Optimization problem
trade.solve(
    quality=config["quality"],
    risk=config["risk"],
    alpha_risk=config["alpha_risk"],
    beta_risk=config["beta_risk"],
)

# Runtime for optimization
runtime_opt = time.time() - start_opt

# Print the revenue
revenue = trade.revenue()

# Save the trade outcomes
P_DA_arr = trade.P_DA_arr
if "FCR-D" in trade.markets:
    c_up_arr = trade.c_up_arr
    c_dn_arr = trade.c_dn_arr
else:
    c_up_arr = np.zeros(len(P_DA_arr))
    c_dn_arr = np.zeros(len(P_DA_arr))
# Shifting the SOC values by 1 datapoint since SOC_t is the state of charge at the end of time t
SOC_plot = np.concatenate((np.full_like(trade.SOC_arr[..., :1], trade.SOC_i), trade.SOC_arr[..., :-1]),axis=-1)
if trade.scheme == "Stochastic":
    SOC_arr = np.mean(SOC_plot, axis=0)
elif trade.scheme == "Deterministic":
    SOC_arr = SOC_plot

# Save trade revenues
rev_FCR_D = trade.rev_FCR_D
rev_DA = trade.rev_DA
rev_IM = trade.rev_IM
PnL = trade.PnL

# Ensure results folder exists
results_dir = Path(f"VT/VT_results/{VT_name}")
results_dir.mkdir(exist_ok=True)

trade_file = results_dir / "trade.csv"
revenue_file = results_dir / "revenue.csv"
runtime_file = results_dir / "runtime.csv"

# Timestamp index (96 intervals)
index_trade = pd.date_range(
    start=f"{date_trade} 00:00",
    end=f"{date_trade} 23:45",
    freq="15min",
)

# TRADE DATAFRAME
trade_df = pd.DataFrame(
    {
        "P_DA": P_DA_arr,
        "c_up": c_up_arr,
        "c_dn": c_dn_arr,
        "SOC": SOC_arr,
    },
    index=index_trade,
)
trade_df.index.name = "timestamp"

# REVENUE DATAFRAME
revenue_df = pd.DataFrame(
    {
        "rev_FCR_D": rev_FCR_D,
        "rev_DA": rev_DA,
        "rev_IM": rev_IM,
        "PnL": PnL,
    },
    index=index_trade,
)
revenue_df.index.name = "timestamp"

# RUNTIME DATAFRAME
runtime_df = pd.DataFrame(
    {
        "data_loading": [runtime_data],
        "optimization": [runtime_opt],
    },
    index=[date_trade],
)
runtime_df.index.name = "date"

# Save results. Append if file exists
if trade_file.exists():
    trade_df.to_csv(trade_file, mode="a", header=False)
else:
    trade_df.to_csv(trade_file)

if revenue_file.exists():
    revenue_df.to_csv(revenue_file, mode="a", header=False)
else:
    revenue_df.to_csv(revenue_file)

if runtime_file.exists():
    runtime_df.to_csv(runtime_file, mode="a", header=False)
else:
    runtime_df.to_csv(runtime_file)

print(f"Runtimes\nData Loading: {runtime_data}\nOptimization: {runtime_opt}")