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
import matplotlib.pyplot as plt
import os

# Initialize the request protocol
CLIENT = EntsoePandasClient(api_key=API_KEY_ENTSOE)

# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("VT_name", type=str)
args = parser.parse_args()
# Store the parsed name of the VT
VT_name = args.VT_name


## Test

# Read config
with open(f"tests/tests_config/test_{VT_name}.yml", "r") as f:
    config = yaml.safe_load(f)

# Date of Trade - Trading for previous day for now since forecasting module is not yet ready
date_trade = previous_day_str(date.today().strftime("%Y-%m-%d"))

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

# Solve the Optimization problem
trade.solve(
    quality=config["quality"],
    risk=config["risk"],
    alpha_risk=config["alpha_risk"],
    beta_risk=config["beta_risk"],
)

# Print the revenue
print("Revenue function:\n")
trade.revenue()

# plot results
fig = trade.plot()
if trade.risk:
    fig_risk = trade.plot_risk()
if os.getenv("CI") != "true":
    plt.show()
    plt.close(fig)
    plt.close(fig_risk)
else:
    plt.close(fig)
    plt.close(fig_risk)


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
