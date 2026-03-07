# Import packages
import argparse
import pandas as pd
from dotenv import load_dotenv
import os
from ortools.linear_solver import pywraplp
from entsoe import EntsoePandasClient
import numpy as np
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import KFold
from requests.exceptions import HTTPError
import yaml
from pathlib import Path
import time

load_dotenv()  # loads .env into environment variables
API_KEY_ENTSOE = os.getenv("API_KEY_ENTSOE")

# Initialize the request protocol
CLIENT = EntsoePandasClient(api_key=API_KEY_ENTSOE)

# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("VT_name", type=str)
args = parser.parse_args()
# Store the parsed name of the VT
VT_name = args.VT_name

# Function to print out next day in string
def next_day_str(date_str: str) -> str:
    """
    Given a date string in 'YYYY-MM-DD' format, return the next day's date string.
    Example: '2025-10-02' -> '2025-10-03'
    """
    return (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

# Function to print out previous day in string
def previous_day_str(date_str: str) -> str:
    """
    Given a date string in 'YYYY-MM-DD' format, return the previous day's date string.
    Example: '2025-10-02' -> '2025-10-01'
    """
    return (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=-1)).strftime("%Y-%m-%d")

# Function to make x-ticks for the plots
def date_hour_formatter(x, pos):
    dt = md.num2date(x)
    if dt.hour == 0:
        return dt.strftime("%d %b")
    return dt.strftime("%H:%M")

# Functions to acces the market data
def getPriceDA(country='FR', start=pd.Timestamp("2025-10-11", tz="Europe/Paris"), end=pd.Timestamp("2025-10-12", tz="Europe/Paris")):
    df = pd.DataFrame(CLIENT.query_day_ahead_prices(
        country_code=country,
        start=start,
        end=end,
    ))
    df.columns = ['day-ahead']
    return df

def getPriceIM(country='FR', start=pd.Timestamp("2025-10-11", tz="Europe/Paris"), end=pd.Timestamp("2025-10-12", tz="Europe/Paris")):
    df = pd.DataFrame(CLIENT.query_imbalance_prices(
        country_code=country,
        start=start,
        end=end,
        psr_type=None,
    ))
    return df

def getPriceFCR_contracted(country='FR', start=pd.Timestamp("2025-10-11", tz="Europe/Paris"), end=pd.Timestamp("2025-10-12", tz="Europe/Paris")):
    df = pd.DataFrame(CLIENT.query_contracted_reserve_prices(
        country_code=country,
        process_type = "A52",
        type_marketagreement_type = "A01",
        start=start,
        end=end,
        psr_type=None,
    ))
    return df["Symmetric"]

def getVolumeFCR_contracted(country='FR', start=pd.Timestamp("2025-10-11", tz="Europe/Paris"), end=pd.Timestamp("2025-10-12", tz="Europe/Paris")):
    df = pd.DataFrame(CLIENT.query_contracted_reserve_amount(
        country_code=country,
        process_type = "A52",
        type_marketagreement_type = "A01",
        start=start,
        end=end,
        psr_type=None,
    ))
    return df["Symmetric"]

def getVolumeFCR_activation(country='FR', start=pd.Timestamp("2025-10-11", tz="Europe/Paris"), end=pd.Timestamp("2025-10-12", tz="Europe/Paris")):
    df = pd.DataFrame(CLIENT.query_activated_balancing_energy(
        country_code=country,
        start=start,
        end=end,
        business_type = "A95",
        psr_type=None,
    ))
    df.columns = df.columns.droplevel()
    return df

def getPriceFCR_activation(country='FR', start=pd.Timestamp("2025-10-11", tz="Europe/Paris"), end=pd.Timestamp("2025-10-12", tz="Europe/Paris")):
    df = pd.DataFrame(CLIENT.query_activated_balancing_energy_prices(
        country_code=country,
        start=start,
        end=end,
        process_type='A16',
        psr_type=None,
        business_type = "A95",
        standard_market_product=None, 
        original_market_product=None,
    ))
    return df

# BESS trading strategy class 
class BESS_trading_strategy:

    def __init__(self, SOC_max: float, R_max: float, SOC_i: float = 0, FCR_max: float = 0.8, 
                 scheme: str = "Deterministic", W: int = 1,
                 markets: str = "FCR-D + DA with imbalance", country: str = "FR", 
                 start_date: str = "2025-10-11", end_date: str = "2025-10-11"):
        self.SOC_max = SOC_max
        self.R_max = R_max
        self.SOC_i = SOC_i
        self.FCR_max = FCR_max
        self.scheme = scheme
        self.W = W
        self.markets = markets
        self.country = country
        self.start_date = start_date
        self.end_date = end_date

        # Initialize solve status
        self.solve_status = "Not solved yet"

        # load in essential parameters
        self.essential_param()

    def essential_param(self):
        # Initializing other parameters
        # FCR-D Prices (EUR/MW)
        df_lambda_FCR_D_up = getPriceFCR_contracted(country = 'FR', start = pd.Timestamp(self.start_date, tz="Europe/Paris"), end = pd.Timestamp(next_day_str(self.end_date), tz="Europe/Paris"))
        df_lambda_FCR_D_dn = getPriceFCR_contracted(country = 'FR', start = pd.Timestamp(self.start_date, tz="Europe/Paris"), end = pd.Timestamp(next_day_str(self.end_date), tz="Europe/Paris"))

        # FCR-D Volume contracted (MW)
        self.df_volcon_FCR_D_up = getVolumeFCR_contracted(country = 'FR', start = pd.Timestamp(self.start_date, tz="Europe/Paris"), end = pd.Timestamp(next_day_str(self.end_date), tz="Europe/Paris"))
        self.df_volcon_FCR_D_dn = getVolumeFCR_contracted(country = 'FR', start = pd.Timestamp(self.start_date, tz="Europe/Paris"), end = pd.Timestamp(next_day_str(self.end_date), tz="Europe/Paris"))

        # FCR-D Volume activated (MW)
        try:
            self.df_volact_FCR_D_up = getVolumeFCR_activation(country = 'FR', start = pd.Timestamp(self.start_date, tz="Europe/Paris"), end = pd.Timestamp(next_day_str(self.end_date), tz="Europe/Paris"))["Up"]
        except HTTPError: 
            print("Activated up volume data cannot be extracted (replacing with 5% of contracted values)")
            self.df_volact_FCR_D_up = 0.05 * self.df_volcon_FCR_D_up
        try:
            self.df_volact_FCR_D_dn = getVolumeFCR_activation(country = 'FR', start = pd.Timestamp(self.start_date, tz="Europe/Paris"), end = pd.Timestamp(next_day_str(self.end_date), tz="Europe/Paris"))["Down"]
        except HTTPError: 
            print("Activated down volume data cannot be extracted (replacing with 5% of contracted values)")
            self.df_volact_FCR_D_dn = 0.05 * self.df_volcon_FCR_D_dn
    
        # Imbalance Prices (EUR/MWh)
        df_lambda_up = getPriceIM(country='FR', start=pd.Timestamp(self.start_date, tz="Europe/Paris"), end=pd.Timestamp(next_day_str(self.end_date), tz="Europe/Paris"))["Short"]
        df_lambda_dn = getPriceIM(country='FR', start=pd.Timestamp(self.start_date, tz="Europe/Paris"), end=pd.Timestamp(next_day_str(self.end_date), tz="Europe/Paris"))["Long"]

        # DA Prices (EUR/MWh)
        df_lambda_DA = getPriceDA(country='FR', start=pd.Timestamp(self.start_date, tz="Europe/Paris"), end=pd.Timestamp(next_day_str(self.end_date), tz="Europe/Paris"))["day-ahead"].loc[self.start_date: self.end_date]

        # Downsample (be linear interpolation) the DA prices to 15-mins (applies to befre 1st Oct 2025)
        df_fixed_list = []
        for day, g in df_lambda_DA.groupby(df_lambda_DA.index.normalize()):
            if len(g) > 1 and (g.index[1] - g.index[0]).seconds / 60 == 60:
                # 1-hour data → resample to 15-min with mid-hour alignment
                g_fixed = (
                    #g.resample("15min").interpolate("linear").shift(freq="30min").reindex(pd.date_range(day, periods=96, freq="15min", tz=g.index.tz), method="nearest")
                    g.resample("15min").ffill().reindex(pd.date_range(day, periods=96, freq="15min", tz=g.index.tz), method="ffill")
                )
                df_fixed_list.append(g_fixed)
            else:
                df_fixed_list.append(g)

        # Concatenate all daily segments back together
        df_lambda_DA = pd.concat(df_fixed_list).sort_index()


        # Big M
        self.M = 100000000
        
        # Number of time steps
        self.T = 96

        # Temporal Rsolution Index
        self.n_FCR = int(60 / (df_lambda_FCR_D_up.index[1] - df_lambda_FCR_D_up.index[0]).seconds * 60)
        self.n_im  = int(60 / (df_lambda_up.index[1] - df_lambda_up.index[0]).seconds * 60)
        self.n_DA  = int(60 / (df_lambda_DA.index[1] - df_lambda_DA.index[0]).seconds * 60)

        # Fold size for k-fold cross validation for ex-post out-of-sampe analysis
        self.k = 0.2 # % of W

        # Get numpy array of ratio of activated volume to the contracted volume
        self.r_up_full = (self.df_volact_FCR_D_up/self.df_volcon_FCR_D_up).to_numpy().ravel()
        self.r_dn_full = (self.df_volact_FCR_D_dn/self.df_volcon_FCR_D_dn).to_numpy().ravel()

        # Convert prices to np array
        self.lambda_FCR_D_up_full = df_lambda_FCR_D_up.to_numpy().ravel()
        self.lambda_FCR_D_dn_full = df_lambda_FCR_D_dn.to_numpy().ravel()
        self.lambda_up_full = df_lambda_up.to_numpy().ravel()
        self.lambda_dn_full = df_lambda_dn.to_numpy().ravel()
        self.lambda_im_full = ((df_lambda_up+df_lambda_dn)/2).to_numpy().ravel()
        self.lambda_DA_full = df_lambda_DA.to_numpy().ravel()

        # Stochastic parameter preparation
        if self.scheme == "Stochastic":
            # Historical Standard Deviations of prices
            self.sd_FCR_D_up = 1
            self.sd_FCR_D_dn = 1
            self.sd_up = 60
            self.sd_dn = 60
            self.sd_DA = 40
            # Historical Standard Deviation of ratio of activation of FCR to contracted
            self.sd_r_up = 0.023
            self.sd_r_dn = 0.023

            # To maintain consistent scenarios
            np.random.seed(42)

            # Stochastic FCR-D Prices (EUR/MW)
            # One random shock per 4-hour block (size = 4*n_FCR), repeated over the block and added to the mean price
            self.lambda_Stc_FCR_D_up_full = self.lambda_FCR_D_up_full + np.repeat(
                self.sd_FCR_D_up * np.random.randn(self.W, int(np.ceil(len(self.lambda_FCR_D_up_full) / (4 * self.n_FCR)))),
                4 * self.n_FCR, axis=1)[:, :len(self.lambda_FCR_D_up_full)]
            self.lambda_Stc_FCR_D_dn_full = self.lambda_FCR_D_dn_full + np.repeat(
                self.sd_FCR_D_dn * np.random.randn(self.W, int(np.ceil(len(self.lambda_FCR_D_dn_full) / (4 * self.n_FCR)))),
                4 * self.n_FCR, axis=1)[:, :len(self.lambda_FCR_D_dn_full)]
            #self.lambda_Stc_FCR_D_up_full = self.lambda_FCR_D_up_full + self.sd_FCR_D_up * np.random.randn(self.W, len(self.lambda_FCR_D_up_full))
            #self.lambda_Stc_FCR_D_dn_full = self.lambda_FCR_D_dn_full + self.sd_FCR_D_dn * np.random.randn(self.W, len(self.lambda_FCR_D_dn_full))

            # Stochastic Imbalance Prices (EUR/MWh)
            self.lambda_Stc_up_full = self.lambda_up_full + self.sd_up * np.random.randn(self.W, len(self.lambda_up_full))
            self.lambda_Stc_dn_full = self.lambda_dn_full + self.sd_dn * np.random.randn(self.W, len(self.lambda_dn_full))
            # The final imbalance (Stochastic)
            self.lambda_Stc_im_full = (self.lambda_Stc_up_full+self.lambda_Stc_dn_full)/2

            # Stochastic DA Prices (EUR/MWh)
            self.lambda_Stc_DA_full = self.lambda_DA_full + self.sd_DA * np.random.randn(self.W, len(self.lambda_DA_full))

            # Stochastic ratio of activated to contracted FCR
            self.r_Stc_up_full = np.clip(np.random.normal(0.025, self.sd_r_up, (self.W, len(self.r_up_full))), 0, 0.20)
            self.r_Stc_dn_full = np.clip(np.random.normal(0.025, self.sd_r_dn, (self.W, len(self.r_up_full))), 0, 0.20)
        
        return

    def solve(self, quality: bool = False, risk: bool = False, alpha_risk: float = 1, beta_risk: float = 0):
        self.quality = quality
        self.risk = risk
        self.alpha_risk = alpha_risk
        self.beta_risk = beta_risk
        self.solve_status = "Normal"

        # Running the correct model for all the dates in the range
        if self.scheme == "Deterministic" or self.W == 1:
            # Correct the no. of scenarios if not already
            self.W = 1 if self.W != 1 else self.W
            # Arrays to save the optimal values
            self.Obj_arr = np.zeros(len(pd.date_range(self.start_date, self.end_date)))
            self.P_DA_arr = np.array([])
            self.P_act_arr = np.array([])
            self.c_up_arr = np.array([])
            self.c_dn_arr = np.array([])
            self.SOC_arr = np.array([])

            # Run for all the dates in the date range
            for i, date_trade in enumerate(pd.date_range(self.start_date, self.end_date)):
                date_trade = date_trade.strftime("%Y-%m-%d")

                # Get numpy array of prices
                self.lambda_FCR_D_up = self.lambda_FCR_D_up_full[i*24*self.n_FCR:(i+1)*24*self.n_FCR]
                self.lambda_FCR_D_dn = self.lambda_FCR_D_dn_full[i*24*self.n_FCR:(i+1)*24*self.n_FCR]
                self.lambda_up = self.lambda_up_full[i*24*self.n_im:(i+1)*24*self.n_im]
                self.lambda_dn = self.lambda_dn_full[i*24*self.n_im:(i+1)*24*self.n_im]
                self.lambda_im = self.lambda_im_full[i*24*self.n_im:(i+1)*24*self.n_im] # The final imbalance
                self.lambda_DA = self.lambda_DA_full[i*24*self.n_DA:(i+1)*24*self.n_DA]
                self.r_up = self.r_up_full[i*24*self.n_im:(i+1)*24*self.n_im]
                self.r_dn = self.r_dn_full[i*24*self.n_im:(i+1)*24*self.n_im]
                
                # Running the model
                if "FCR-D" in self.markets:
                    self.Obj_arr[i], P_DA_o, P_act_o, c_up_o, c_dn_o, SOC_o = self.FCRD_DA_IMB_daily(date_trade)
                    self.c_up_arr = np.append(self.c_up_arr, c_up_o)
                    self.c_dn_arr = np.append(self.c_dn_arr, c_dn_o)
                elif self.markets == "DA with imbalance":
                    self.Obj_arr[i], P_DA_o, P_act_o, SOC_o = self.FCRD_DA_IMB_daily(date_trade)
                self.P_DA_arr =  np.append(self.P_DA_arr, P_DA_o)
                self.P_act_arr = np.append(self.P_act_arr, P_act_o)
                self.SOC_arr = np.append(self.SOC_arr, SOC_o)

            # Return
            if self.solve_status == "Normal":
                print("Solving finished successfully")
                return
            else:
                print("!!!Solution not found!!!")
                return

        elif self.scheme == "Stochastic":
            # Arrays to save the optimal values
            self.Obj_arr = np.zeros(len(pd.date_range(self.start_date, self.end_date)))
            self.zeta_arr = np.array([])
            self.nu_arr = np.empty((len(pd.date_range(self.start_date, self.end_date)), 0))
            self.P_DA_arr = np.array([])
            self.c_up_arr = np.array([])
            self.c_dn_arr = np.array([])
            if self.quality:
                self.P_act_arr = np.empty((int(self.k*self.W), 0))
                self.SOC_arr = np.empty((int(self.k*self.W), 0))
            else:
                self.P_act_arr = np.empty((self.W, 0))
                self.SOC_arr = np.empty((self.W, 0))

            # Run for all the dates in the date range
            for i, date_trade in enumerate(pd.date_range(self.start_date, self.end_date)):
                date_trade = date_trade.strftime("%Y-%m-%d")

                # Get numpy array of stochastic prices
                self.lambda_Stc_FCR_D_up = self.lambda_Stc_FCR_D_up_full[:,i*24*self.n_FCR:(i+1)*24*self.n_FCR]
                self.lambda_Stc_FCR_D_dn = self.lambda_Stc_FCR_D_dn_full[:,i*24*self.n_FCR:(i+1)*24*self.n_FCR]
                self.lambda_Stc_up = self.lambda_Stc_up_full[:,i*24*self.n_im:(i+1)*24*self.n_im]
                self.lambda_Stc_dn = self.lambda_Stc_dn_full[:,i*24*self.n_im:(i+1)*24*self.n_im]
                self.lambda_Stc_im = self.lambda_Stc_im_full[:,i*24*self.n_im:(i+1)*24*self.n_im] # The final imbalance
                self.lambda_Stc_DA = self.lambda_Stc_DA_full[:,i*24*self.n_DA:(i+1)*24*self.n_DA]
                self.r_Stc_up = self.r_Stc_up_full[:,i*24*self.n_im:(i+1)*24*self.n_im]
                self.r_Stc_dn = self.r_Stc_dn_full[:,i*24*self.n_im:(i+1)*24*self.n_im]

                # Running the model
                if "FCR-D" in self.markets:
                    if self.quality:
                        self.Obj_arr[i], zeta_o, nu_o, P_DA_o, P_act_o, c_up_o, c_dn_o, SOC_o = self.cross_validate_STC_FCRD_DA_IMB(date_trade)
                    else:
                        self.Obj_arr[i], zeta_o, nu_o, P_DA_o, P_act_o, c_up_o, c_dn_o, SOC_o = self.STC_FCRD_DA_IMB_daily(date_trade, self.W, self.lambda_Stc_FCR_D_up, self.lambda_Stc_FCR_D_dn, self.lambda_Stc_DA, self.lambda_Stc_im, self.r_Stc_up, self.r_Stc_dn)
                    self.c_up_arr = np.append(self.c_up_arr, c_up_o)
                    self.c_dn_arr = np.append(self.c_dn_arr, c_dn_o)
                elif self.markets == "DA with imbalance":
                    if self.quality:
                        self.Obj_arr[i], zeta_o, nu_o, P_DA_o, P_act_o, SOC_o = self.cross_validate_STC_FCRD_DA_IMB(date_trade)
                    else:
                        self.Obj_arr[i], zeta_o, nu_o, P_DA_o, P_act_o, SOC_o = self.STC_FCRD_DA_IMB_daily(date_trade, self.W, self.lambda_Stc_FCR_D_up, self.lambda_Stc_FCR_D_dn, self.lambda_Stc_DA, self.lambda_Stc_im, self.r_Stc_up, self.r_Stc_dn)
                self.zeta_arr =  np.append(self.zeta_arr, zeta_o)
                if self.quality:
                    self.nu_arr = np.hstack((self.nu_arr, nu_o.reshape(1,int(self.k*self.W))))
                else:
                    self.nu_arr = np.hstack((self.nu_arr, nu_o.reshape(1,self.W)))
                self.P_DA_arr =  np.append(self.P_DA_arr, P_DA_o)
                self.P_act_arr = np.hstack((self.P_act_arr, P_act_o))
                self.SOC_arr = np.hstack((self.SOC_arr, SOC_o))

            if self.solve_status == "Normal":
                print("Solving finished successfully")
                return
            else:
                print("!!!Solution not found!!!")
                return

    def cross_validate_STC_FCRD_DA_IMB(self, date_trade):
        self.fold_size = int(self.k * self.W)
        kf = KFold(n_splits=self.W // self.fold_size, shuffle=True, random_state=42)

        # Initialize the in and out sample gap
        best_gap = float("inf")

        for out_idx, in_idx in kf.split(range(self.W)):
            # Split seen (in-sample) and unseen (out-of-sample) scenarios
            self.lambda_Stc_FCR_D_up_in, self.lambda_Stc_FCR_D_dn_in, self.lambda_Stc_im_in, self.lambda_Stc_DA_in, self.r_Stc_up_in, self.r_Stc_dn_in = self.lambda_Stc_FCR_D_up[in_idx], self.lambda_Stc_FCR_D_dn[in_idx], self.lambda_Stc_im[in_idx], self.lambda_Stc_DA[in_idx], self.r_Stc_up[in_idx], self.r_Stc_dn[in_idx]
            self.lambda_Stc_FCR_D_up_out, self.lambda_Stc_FCR_D_dn_out, self.lambda_Stc_im_out, self.lambda_Stc_DA_out, self.r_Stc_up_out, self.r_Stc_dn_out = self.lambda_Stc_FCR_D_up[out_idx], self.lambda_Stc_FCR_D_dn[out_idx], self.lambda_Stc_im[out_idx], self.lambda_Stc_DA[out_idx], self.r_Stc_up[out_idx], self.r_Stc_dn[out_idx]

            # Solve optimization on seen (or insample) scenarios
            if "FCR-D" in self.markets:
                Obj_exp_in, zeta_in_o, nu_in_o, P_DA_in_o, P_act_in_o, c_up_in_o, c_dn_in_o, SOC_in_o = self.STC_FCRD_DA_IMB_daily(date_trade, self.fold_size, self.lambda_Stc_FCR_D_up_in, self.lambda_Stc_FCR_D_dn_in, self.lambda_Stc_DA_in, self.lambda_Stc_im_in, self.r_Stc_up_in, self.r_Stc_dn_in)
            elif self.markets == "DA with imbalance":
                Obj_exp_in, zeta_in_o, nu_in_o, P_DA_in_o, P_act_in_o, SOC_in_o = self.STC_FCRD_DA_IMB_daily(date_trade, self.fold_size, self.lambda_Stc_FCR_D_up_in, self.lambda_Stc_FCR_D_dn_in, self.lambda_Stc_DA_in, self.lambda_Stc_im_in, self.r_Stc_up_in, self.r_Stc_dn_in)

            # Compute out-of-sample average revenue
            if "FCR-D" in self.markets:
                Obj_avg_out = (1 / ((1-self.k) * self.W)) * np.sum(self.lambda_Stc_FCR_D_up_out * c_up_in_o
                                                                + self.lambda_Stc_FCR_D_dn_out * c_dn_in_o
                                                                + self.lambda_Stc_DA_out * (P_DA_in_o / self.n_DA)
                                                                + self.lambda_Stc_im_out * ((P_act_in_o.mean(axis=0) - P_DA_in_o) / self.n_im))
            elif self.markets == "DA with imbalance":
                Obj_avg_out = (1 / ((1-self.k) * self.W)) * np.sum(self.lambda_Stc_DA_out * (P_DA_in_o / self.n_DA)
                                                                + self.lambda_Stc_im_out * ((P_act_in_o.mean(axis=0) - P_DA_in_o) / self.n_im))

            # Compare performance
            gap = abs(Obj_avg_out - Obj_exp_in)
            if gap < best_gap:
                best_gap = gap
                Obj_o = Obj_exp_in
                zeta_o = zeta_in_o
                nu_o = nu_in_o
                P_DA_o = P_DA_in_o
                P_act_o = P_act_in_o
                if "FCR-D" in self.markets:
                    c_up_o = c_up_in_o
                    c_dn_o = c_dn_in_o
                SOC_o = SOC_in_o
                self.gap_KF_OSA = gap*100/Obj_exp_in

        # Return values
        if "FCR-D" in self.markets:
            return Obj_o, zeta_o, nu_o, P_DA_o, P_act_o, c_up_o, c_dn_o, SOC_o
        elif self.markets == "DA with imbalance":
            return Obj_o, zeta_o, nu_o, P_DA_o, P_act_o, SOC_o

    def FCRD_DA_IMB_daily(self, date_trade):
        # Initialize the solver
        solver = pywraplp.Solver.CreateSolver("SCIP") # GLOP, CBC, SCIP, SAT, GUROBI, CP-SAT

        # Decision variables
        P_DA = [solver.NumVar(-self.R_max, self.R_max, f"P_DA_{t}") for t in range(self.T)]
        P_act = [solver.NumVar(-self.R_max, self.R_max, f"P_act_{t}") for t in range(self.T)]
        if "FCR-D" in self.markets:
            c_up = [solver.NumVar(0, solver.infinity(), f"c_up_{t}") for t in range(self.T)]
            c_dn = [solver.NumVar(0, solver.infinity(), f"c_dn_{t}") for t in range(self.T)]
            #y_up = [solver.IntVar(0, 1, f"y_up_{t}") for t in range(T)]
            #y_dn = [solver.IntVar(0, 1, f"y_dn_{t}") for t in range(T)]
        SOC = [solver.NumVar(0, self.SOC_max, f"SOC_{t}") for t in range(self.T)]

        # Objective function
        if "FCR-D" in self.markets:
            solver.Maximize(solver.Sum( ( self.lambda_FCR_D_up[t]*c_up[t] 
                                        + self.lambda_FCR_D_dn[t]*c_dn[t] 
                                        + self.lambda_DA[t]*P_DA[t]/self.n_DA 
                                        + self.lambda_im[t]*(P_act[t]-P_DA[t])/self.n_im  ) for t in range(self.T)))
            
        elif self.markets == "DA with imbalance":
            solver.Maximize(solver.Sum( ( self.lambda_DA[t]*P_DA[t]/self.n_DA 
                                        + self.lambda_im[t]*(P_act[t]-P_DA[t])/self.n_im  ) for t in range(self.T)))

        for t in range(self.T):

            if "FCR-D" in self.markets:
                # FCR-D violation
                #solver.Add( solver.Sum(y_up[t] for w in range(self.W)) <= self.q*self.W )
                #solver.Add( solver.Sum(y_dn[t] for w in range(self.W)) <= self.q*self.W )

                # FCR bidding limit in terms of SOC usage
                solver.Add( (c_up[t]+c_dn[t])/self.n_FCR <= self.FCR_max*self.SOC_max )

                # 6 block bids for FCR in EFA
                if t in range(0,self.T,round(self.T/6)):
                    for j in range(t+1,t+round(self.T/6)):
                        solver.Add(c_up[t] == c_up[j])
                        solver.Add(c_dn[t] == c_dn[j])

            for w in range(self.W):
                
                # State of Charge and FCR-D power restrictions
                if t==0:
                    solver.Add(SOC[t] == self.SOC_i - (P_act[t])/self.n_DA)
                    if "FCR-D" in self.markets:
                        solver.Add(c_up[t]/self.n_FCR <= self.SOC_i)
                        solver.Add(c_dn[t]/self.n_FCR <= self.SOC_max - self.SOC_i)
                else:
                    solver.Add(SOC[t] == SOC[t-1] - (P_act[t])/self.n_DA)
                    if "FCR-D" in self.markets:
                        solver.Add(c_up[t]/self.n_FCR <= SOC[t-1])
                        solver.Add(c_dn[t]/self.n_FCR <= self.SOC_max - SOC[t-1])
                
                # Actual Power limits
                if "FCR-D" in self.markets:
                    #solver.Add(P_act[t] + self.R_max - c_dn[t] >= y_dn[t]*self.M)
                    solver.Add(P_act[t] + self.R_max - self.r_dn[t]*c_dn[t] >= 0)
                    #solver.Add(P_act[t] - self.R_max + c_up[t] <= y_up[t]*self.M)
                    solver.Add(P_act[t] - self.R_max + self.r_up[t]*c_up[t] <= 0)

        # Solve the problem
        status = solver.Solve()
        if status == pywraplp.Solver.INFEASIBLE:
            print(date_trade, "Problem is infeasible. Stopping early.")  # noqa: T201
            self.solve_status = "Infeasible"
        if status != pywraplp.Solver.OPTIMAL:
            print(date_trade, ": Not optimal")
            self.solve_status = "Not Optimal"

        #Save Optimal values of this sub-problem
        P_DA_o = np.array([P_DA[t].solution_value() for t in range(self.T)])  # noqa: N806
        P_act_o = np.array([P_act[t].solution_value() for t in range(self.T)])  # noqa: N806
        if "FCR-D" in self.markets:
            c_up_o = np.array([c_up[t].solution_value() for t in range(self.T)])  # noqa: N806
            c_dn_o = np.array([c_dn[t].solution_value() for t in range(self.T)])  # noqa: N806
            #y_up_o = np.array([y_up[t].solution_value() for t in range(self.T)])  # noqa: N806
            #y_dn_o = np.array([y_dn[t].solution_value() for t in range(self.T)])  # noqa: N806
        SOC_o = np.array([SOC[t].solution_value() for t in range(self.T)])  # noqa: N806

        # Return values
        if "FCR-D" in self.markets:
            return solver.Objective().Value(), P_DA_o, P_act_o, c_up_o, c_dn_o, SOC_o
        elif self.markets == "DA with imbalance":
            return solver.Objective().Value(), P_DA_o, P_act_o, SOC_o
    
    def STC_FCRD_DA_IMB_daily(self, date_trade, W_local, lambda_Stc_FCR_D_up_local, lambda_Stc_FCR_D_dn_local, lambda_Stc_DA_local, lambda_Stc_im_local, r_Stc_up_local, r_Stc_dn_local):
        # Model Design

        # Initialize the solver
        solver = pywraplp.Solver.CreateSolver("SCIP") # GLOP, CBC, SCIP, SAT, GUROBI, CP-SAT

        # Decision variables
        P_DA = [solver.NumVar(-self.R_max, self.R_max, f"P_DA_{t}") for t in range(self.T)]
        P_act = [[solver.NumVar(-self.R_max, self.R_max, f"P_act[{t},{w}]") for w in range(W_local)] for t in range(self.T)]
        if "FCR-D" in self.markets:
            c_up = [solver.NumVar(0, 2*self.R_max, f"c_up_{t}") for t in range(self.T)]
            c_dn = [solver.NumVar(0, 2*self.R_max, f"c_dn_{t}") for t in range(self.T)]
            #y_up = [[solver.IntVar(0, 1, f"y_up[{t},{w}]") for w in range(W_local)] for t in range(self.T)]
            #y_dn = [[solver.IntVar(0, 1, f"y_dn[{t},{w}]") for w in range(W_local)] for t in range(self.T)]
        SOC = [[solver.NumVar(0, self.SOC_max, f"SOC[{t},{w}]") for w in range(W_local)] for t in range(self.T)]
        z_rn = [solver.NumVar(-solver.infinity(), solver.infinity(), f"z_rn_{w}") for w in range(W_local)] # Risk neutral revenue function
        zeta = solver.NumVar(-solver.infinity(), solver.infinity(), "zeta") # Value-at-Risk (VaR)
        theta = [solver.IntVar(0, 1, f"theta_{w}") for w in range(W_local)] # auxillary binary variable for VaR calculation
        nu = [solver.NumVar(0, solver.infinity(), f"nu_{w}") for w in range(W_local)] # shortfall from VaR at each scenario (will only take non-negative values)


        # Objective function
        if not self.risk:
            solver.Maximize(
                (1/W_local) * solver.Sum(z_rn[w] for w in range(W_local))  # Equiprobable scenarios
            )
        else:
            solver.Maximize(
                (1 - self.beta_risk) * (1/W_local) * solver.Sum(z_rn[w] for w in range(W_local))  # Equiprobable scenarios
                + self.beta_risk * ( zeta - (1/(1-self.alpha_risk)) * (1/W_local) * solver.Sum(nu[w] for w in range(W_local)) )  # CVaR 
            )
        
        # Percentile setting for VaR
        solver.Add( (1/W_local) * solver.Sum(theta[w] for w in range(W_local)) <= 1 - self.alpha_risk )

        for w in range(W_local):
            # Risk Neutral scenario-wise revenue
            if "FCR-D" in self.markets:
                solver.Add(z_rn[w] == solver.Sum((
                                            lambda_Stc_FCR_D_up_local[w,t] * c_up[t]
                                            + lambda_Stc_FCR_D_dn_local[w,t] * c_dn[t]
                                            + lambda_Stc_DA_local[w,t] * P_DA[t] / self.n_DA
                                            + lambda_Stc_im_local[w,t] * (P_act[t][w] - P_DA[t]) / self.n_im
                                        ) for t in range(self.T)
                                    )
                                )
            elif self.markets == "DA with imbalance":
                solver.Add(z_rn[w] == solver.Sum((
                                            lambda_Stc_DA_local[w,t] * P_DA[t] / self.n_DA
                                            + lambda_Stc_im_local[w,t] * (P_act[t][w] - P_DA[t]) / self.n_im
                                        ) for t in range(self.T)
                                    )
                                )

            # VaR calculation by limits
            solver.Add( zeta <= z_rn[w] + self.M*theta[w] )
            
            # Shortfall from VaR in each scenario
            solver.Add( nu[w] >= zeta - z_rn[w] )
        
        for t in range(self.T):

            if "FCR-D" in self.markets:
                # FCR-D violation
                #solver.Add( solver.Sum(y_up[t][w] for w in range(W_local)) <= self.q*W_local )
                #solver.Add( solver.Sum(y_dn[t][w] for w in range(W_local)) <= self.q*W_local )
                
                # FCR bidding limit in terms of SOC usage
                solver.Add( (c_up[t]+c_dn[t])/self.n_FCR <= self.FCR_max*self.SOC_max)
            
                # 6 block bids for FCR in EFA
                if t in range(0,self.T,round(self.T/6)):
                    for j in range(t+1,t+round(self.T/6)):
                        solver.Add(c_up[t] == c_up[j])
                        solver.Add(c_dn[t] == c_dn[j])

            for w in range(W_local):

                # State of Charge and FCR-D power restrictions
                if t==0:
                    solver.Add(SOC[t][w] == self.SOC_i - (P_act[t][w])/self.n_DA)
                    if "FCR-D" in self.markets:
                        solver.Add(c_up[t]/self.n_FCR <= self.SOC_i)
                        solver.Add(c_dn[t]/self.n_FCR <= self.SOC_max - self.SOC_i)
                else:
                    solver.Add(SOC[t][w] == SOC[t-1][w] - (P_act[t][w])/self.n_DA)
                    if "FCR-D" in self.markets:
                        solver.Add(c_up[t]/self.n_FCR <= SOC[t-1][w])
                        solver.Add(c_dn[t]/self.n_FCR <= self.SOC_max - SOC[t-1][w])

                # Actual Power limits
                if "FCR-D" in self.markets:
                    #solver.Add(P_act[t][w] + self.R_max - c_dn[t] >= y_dn[t][w]*self.M)
                    solver.Add(P_act[t][w] + self.R_max - r_Stc_dn_local[w,t]*c_dn[t] >= 0)
                    #solver.Add(P_act[t][w] - self.R_max + c_up[t] <= y_up[t][w]*self.M)
                    solver.Add(P_act[t][w] - self.R_max + r_Stc_up_local[w,t]*c_up[t] <= 0)

        # Solve the problem
        status = solver.Solve()
        if status == pywraplp.Solver.INFEASIBLE:
            print(date_trade, "Problem is infeasible. Stopping early.")  # noqa: T201
            self.solve_status = "Infeasible"
        if status != pywraplp.Solver.OPTIMAL:
            print(date_trade, ": Not optimal")
            self.solve_status = "Not Optimal"

        #Save Optimal values of this sub-problem
        P_DA_o = np.array([P_DA[t].solution_value() for t in range(self.T)])  # noqa: N806
        P_act_o = np.array([[P_act[t][w].solution_value() for w in range(W_local)] for t in range(self.T)]).T  # noqa: N806
        if "FCR-D" in self.markets:
            c_up_o = np.array([c_up[t].solution_value() for t in range(self.T)])  # noqa: N806
            c_dn_o = np.array([c_dn[t].solution_value() for t in range(self.T)])  # noqa: N806
            #y_up_o = np.array([[y_up[t][w].solution_value() for w in range(W_local)] for t in range(self.T)]).T  # noqa: N806
            #y_dn_o = np.array([[y_dn[t][w].solution_value() for w in range(W_local)] for t in range(self.T)]).T  # noqa: N806
        SOC_o = np.array([[SOC[t][w].solution_value() for w in range(W_local)] for t in range(self.T)]).T  # noqa: N80
        zeta_o = zeta.solution_value()
        nu_o = np.array([nu[w].solution_value() for w in range(W_local)])  # noqa: N806
    
        #print("VaR:", zeta_o)
        self.solver = solver
        # Return values
        if "FCR-D" in self.markets:
            return solver.Objective().Value(), zeta_o, nu_o, P_DA_o, P_act_o, c_up_o, c_dn_o, SOC_o
        elif self.markets == "DA with imbalance":
            return solver.Objective().Value(), zeta_o, nu_o, P_DA_o, P_act_o, SOC_o

    def revenue(self):
        if self.solve_status == "Not solved yet":
            return print("Problem not solved yet: run the optimization problem using .solve()")
        # Print total revenue
        if self.scheme == "Deterministic" or self.W==1:
            # Generate individual revenues assuming equiprobable scenarios
            if "FCR-D" in self.markets:
                self.rev_FCR_D = self.lambda_FCR_D_up_full * self.c_up_arr + self.lambda_FCR_D_dn_full * self.c_dn_arr
            else:
                self.rev_FCR_D = np.zeros(len(self.P_DA_arr))
            self.rev_DA = self.lambda_DA_full * (self.P_DA_arr / self.n_DA)
            self.rev_IM = self.lambda_im_full * ((self.P_act_arr - self.P_DA_arr) / self.n_im)
            self.PnL = self.rev_FCR_D +self.rev_DA + self.rev_IM

            return print(f"Revenue: €{np.sum(self.Obj_arr):,.2f}")
        elif self.scheme == "Stochastic":
            # Generate individual revenues assuming equiprobable scenarios
            if "FCR-D" in self.markets:
                self.rev_FCR_D = self.lambda_Stc_FCR_D_up_full.mean(axis=0) * self.c_up_arr + self.lambda_Stc_FCR_D_dn_full.mean(axis=0) * self.c_dn_arr
            else:
                self.rev_FCR_D = np.zeros(len(self.P_DA_arr))
            self.rev_DA = self.lambda_Stc_DA_full.mean(axis=0) * (self.P_DA_arr / self.n_DA)
            if self.quality:
                self.rev_IM = self.lambda_Stc_im_full.mean(axis=0) * ((self.P_act_arr.mean(axis=0) - self.P_DA_arr) / self.n_im)
                #self.rev_IM = (self.lambda_Stc_im_full * ((self.P_act_arr - self.P_DA_arr) / self.n_im)).mean(axis=0)
            else:
                self.rev_IM = (self.lambda_Stc_im_full * ((self.P_act_arr - self.P_DA_arr) / self.n_im)).mean(axis=0)
            self.PnL = self.rev_FCR_D +self.rev_DA + self.rev_IM
            if self.quality:
                return print(f"Potential Revenue: €{np.sum(self.Obj_arr):,.2f} with a gap of {self.gap_KF_OSA:.2f}% between in and out samples")
            else:
                return print(f"Potential Revenue: €{np.sum(self.Obj_arr):,.2f}")
    
    def plot(self):
        if self.solve_status == "Not solved yet":
            return print("Problem not solved yet: run the optimization problem using .solve()")
        # Generate time index
        index_trade = pd.date_range(
            start=f"{self.start_date} 00:00",
            end=f"{self.end_date} 23:45",
            freq="15min",
        )

        # Shifting the SOC values by 1 datapoint since SOC_t is the state of charge at the end of time t
        SOC_plot = np.concatenate((np.full_like(self.SOC_arr[..., :1], self.SOC_i), self.SOC_arr[..., :-1]),axis=-1)

        if self.scheme == "Stochastic" and self.W>1:
            # Compute expected values (means) and uncertainty bands
            P_act_smean = np.mean(self.P_act_arr, axis=0)
            P_act_smin  =  np.min(self.P_act_arr, axis=0)
            P_act_smax  =  np.max(self.P_act_arr, axis=0)
            SOC_smean   =   np.mean(SOC_plot, axis=0)
            SOC_smin    =    np.min(SOC_plot, axis=0)
            SOC_smax    =    np.max(SOC_plot, axis=0)
            lambda_DA_smean = np.mean(self.lambda_Stc_DA_full, axis=0)
            lambda_DA_smin  =  np.min(self.lambda_Stc_DA_full, axis=0)
            lambda_DA_smax  =  np.max(self.lambda_Stc_DA_full, axis=0)
            lambda_im_smean = np.mean(self.lambda_Stc_im_full, axis=0)
            lambda_im_smin  =  np.min(self.lambda_Stc_im_full, axis=0)
            lambda_im_smax  =  np.max(self.lambda_Stc_im_full, axis=0)
            lambda_FCR_D_up_smean = np.mean(self.lambda_Stc_FCR_D_up_full, axis=0)
            lambda_FCR_D_up_smin  =  np.min(self.lambda_Stc_FCR_D_up_full, axis=0)
            lambda_FCR_D_up_smax  =  np.max(self.lambda_Stc_FCR_D_up_full, axis=0)
            lambda_FCR_D_dn_smean = np.mean(self.lambda_Stc_FCR_D_dn_full, axis=0)
            lambda_FCR_D_dn_smin  =  np.min(self.lambda_Stc_FCR_D_dn_full, axis=0)
            lambda_FCR_D_dn_smax  =  np.max(self.lambda_Stc_FCR_D_dn_full, axis=0)

        # PLot results
        plt.figure(figsize=(12, 15))

        # Plot 1: Power Variables
        ax_left = plt.subplot(5, 1, 1)
        # Left axis: €/MWh
        if self.scheme == "Stochastic" and self.W > 1:
            ax_left.plot(index_trade, lambda_DA_smean, label="DA-mean", color="black",drawstyle="steps-post", linewidth=2)
            ax_left.plot(index_trade, lambda_im_smean, label="Imbalance-mean", color="brown",drawstyle="steps-post", linewidth=2)
            ax_left.fill_between(index_trade, lambda_DA_smin, lambda_DA_smax,color="black", alpha=0.2, step="post", label="DA-range")
            ax_left.fill_between(index_trade, lambda_im_smin, lambda_im_smax,color="tab:brown", alpha=0.2, step="post", label="Imbalance-range")
        elif self.scheme == "Deterministic" or self.W == 1:
            ax_left.plot(index_trade, self.lambda_DA_full, label="DA", color="black", drawstyle="steps-post", linewidth=2)
            ax_left.plot(index_trade, self.lambda_im_full, label="Imbalance", color="brown", drawstyle="steps-post", linewidth=2)
        ax_left.set_ylabel("Price (€/MWh)")
        ax_left.grid(True)
        # Right axis: €/MW (FCR)
        if "FCR-D" in self.markets:
            ax_right = ax_left.twinx()   # second y-axis for FCR prices
            if self.scheme == "Stochastic" and self.W > 1:
                ax_right.plot(index_trade, lambda_FCR_D_up_smean, label="FCR-D-up-mean",color="green", drawstyle="steps-post", linewidth=2)
                ax_right.plot(index_trade, lambda_FCR_D_dn_smean, label="FCR-D-dn-mean",color="red", drawstyle="steps-post", linewidth=2)
                #ax_right.fill_between(index_trade, lambda_FCR_D_up_smin, lambda_FCR_D_up_smax, color="tab:green", alpha=0.2, step="post", label="FCR-D-up-range")
                #ax_right.fill_between(index_trade, lambda_FCR_D_dn_smin, lambda_FCR_D_dn_smax, color="tab:red", alpha=0.2, step="post", label="FCR-D-dn-range")
            else:
                ax_right.plot(index_trade, self.lambda_FCR_D_up_full, label="FCR-D-up", color="green", drawstyle="steps-post", linewidth=2)
                ax_right.plot(index_trade, self.lambda_FCR_D_dn_full, label="FCR-D-dn", color="red", drawstyle="steps-post", linewidth=2)
            ax_right.set_ylabel("Contracted price (€/MW)")
            ax_right.legend(loc="upper right")
        ax_left.set_title("Price forecasts")
        ax_left.legend(loc="upper left")
        ax_left.xaxis.set_major_locator(md.HourLocator(interval=2))
        ax_left.xaxis.set_major_formatter(FuncFormatter(date_hour_formatter))


        # Plot 2: Power Variables
        ax = plt.subplot(5, 1, 2)
        plt.plot(index_trade, self.P_DA_arr, label="Day-ahead bid", color="black", drawstyle="steps-post", linewidth=2)
        if self.scheme == "Stochastic" and self.W>1:
            plt.plot(index_trade, P_act_smean, label="Scenario Mean of Actual Position", color="tab:blue", drawstyle="steps-post", linestyle="--")
            # Fill scenario range (uncertainty band)
            plt.fill_between(index_trade, P_act_smin, P_act_smax, color="tab:blue", alpha=0.2, step="post", label="Scenario range")
            plt.title("Bids - Day Ahead")
        elif self.scheme == "Deterministic" or self.W==1:
            plt.plot(index_trade, self.P_act_arr, label="Actual Position", color="tab:blue", drawstyle="steps-post", linestyle="--")
            plt.title("Bids - Day Ahead")
        plt.ylabel("Power (MW)")
        plt.legend()
        plt.grid(True)
        ax.xaxis.set_major_locator(md.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(FuncFormatter(date_hour_formatter))

        
        # Plot 3: Up/Down Reserves (Expected Values)
        if "FCR-D" in self.markets:
            ax = plt.subplot(5, 1, 3)
            plt.plot(index_trade, self.c_up_arr, label="FCR up bid", drawstyle="steps-post", color="tab:green")
            plt.plot(index_trade, self.c_dn_arr, label="FCR down bid", drawstyle="steps-post", color="tab:red")
            plt.title("Bids - Reserve Capacities")
            plt.ylabel("Capacity (MW)")
            plt.legend()
            plt.grid(True)
            ax.xaxis.set_major_locator(md.HourLocator(interval=2))
            ax.xaxis.set_major_formatter(FuncFormatter(date_hour_formatter))

        # Plot 4: State of Charge (SOC)
        if "FCR-D" in self.markets:
            ax = plt.subplot(5, 1, 4)
            # Hard limits
            plt.fill_between(index_trade, 0, self.c_up_arr/4, color="black", alpha=0.5, label="Hard limits")
            plt.fill_between(index_trade, self.SOC_max - self.c_dn_arr/4, self.SOC_max, color="black", alpha=0.5)
        elif self.markets == "DA with imbalance":
            ax = plt.subplot(5, 1, 3)
        if self.scheme == "Stochastic" and self.W>1:
            plt.plot(index_trade, SOC_smean, label="SOC (mean)", color="purple")
            # Fill scenario range (uncertainty band)
            plt.fill_between(index_trade, SOC_smin, SOC_smax, color="purple", alpha=0.2, label="Scenario range")
            plt.title("State of Charge (SOC) Across Scenarios")
        elif self.scheme == "Deterministic" or self.W==1:
            plt.plot(index_trade, SOC_plot, label="SOC", color="purple")
            plt.title("State of Charge (SOC)")
        plt.xlabel("Time")
        plt.ylabel("Energy Level (MWh)")
        plt.legend()
        plt.grid(True)
        ax.xaxis.set_major_locator(md.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(FuncFormatter(date_hour_formatter))

        # Plot 5: PnL plot
        if "FCR-D" in self.markets:
            ax = plt.subplot(5, 1, 5)
        elif self.markets == "DA with imbalance":
            ax = plt.subplot(5, 1, 4)
        # Stack revenues with correct handling of positives and negatives
        bottom_pos = np.zeros(len(index_trade))
        bottom_neg = np.zeros(len(index_trade))
        for rev, label, color in [
            (self.rev_DA, "DA revenue", "black"),
            (self.rev_IM, "Imbalance revenue", "tab:orange"),
            (self.rev_FCR_D if "FCR-D" in self.markets else np.zeros(len(index_trade)), "FCR-D revenue", "tab:red"),
        ]:
            pos = np.where(rev > 0, rev, 0)
            neg = np.where(rev < 0, rev, 0)
            ax.bar(index_trade, pos, bottom=bottom_pos, label=label, color=color, width=0.005)
            ax.bar(index_trade, neg, bottom=bottom_neg, color=color, width=0.005)
            bottom_pos += pos
            bottom_neg += neg
        # Overlay total PnL as a line
        ax.plot(index_trade, self.PnL, color="green", linewidth=2, label="P&L")
        # Secondary axis for cumulative PnL
        ax_cum = ax.twinx()
        ax_cum.plot(index_trade, np.cumsum(self.PnL), color="red", linestyle=":", linewidth=2, label="Cumulative P&L")
        ax_cum.set_ylabel("Cumulative P&L (€)")
        ax.set_ylabel("Revenue (€)")
        ax.set_title("Revenue decomposition and total P&L")
        # Align zeros of both the y-axes while keeping the same upper limit on right axis
        yl = ax.get_ylim()
        yr = ax_cum.get_ylim()
        # position of zero in left axis (as fraction)
        zero_pos = -yl[0] / (yl[1] - yl[0])
        # keep upper limit fixed, shift only lower limit
        ax_cum.set_ylim(yr[1] - (yr[1] / (1 - zero_pos)),yr[1])
        ax.legend(loc="upper left")
        ax_cum.legend(loc="right")
        ax.grid(True)
        ax.xaxis.set_major_locator(md.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(FuncFormatter(date_hour_formatter))
        
        # Show plot
        plt.tight_layout()
        #plt.savefig("result_illustration_example", dpi=500)
        return plt.gcf()
    
    def risk_analysis(self):
        if not self.risk or self.beta_risk == 0:
            return print("Risk neutral mode is selected in the parameters, change it to risk aware mode to run this function.")
        if self.solve_status == "Not solved yet":
            return print("Problem not solved yet: run the optimization problem using .solve()")
        
        beta_risk_original = self.beta_risk

        self.beta_arr = np.arange(0,1.1,0.1)
        rev_FCR_D_ra = np.zeros(len(self.beta_arr))
        rev_DA_ra = np.zeros(len(self.beta_arr))
        rev_IM_ra = np.zeros(len(self.beta_arr)) 
        self.PnL_ra = np.zeros(len(self.beta_arr))
        self.CVaR_arr = np.zeros(len(self.beta_arr))
        
        for i,beta in enumerate(self.beta_arr):
            
            self.beta_risk = beta

            # Arrays to save the optimal values
            zeta_arr_ra = np.array([])
            nu_arr_ra = np.empty((len(pd.date_range(self.start_date, self.end_date)), 0))
            P_DA_arr_ra = np.array([])
            c_up_arr_ra = np.array([])
            c_dn_arr_ra = np.array([])
            if self.quality:
                P_act_arr_ra = np.empty((int(self.k*self.W), 0))
                SOC_arr_ra = np.empty((int(self.k*self.W), 0))
            else:
                P_act_arr_ra = np.empty((self.W, 0))
                SOC_arr_ra = np.empty((self.W, 0))
            
            # Run for all the dates in the date range
            for date_trade in pd.date_range(self.start_date, self.end_date):
                date_trade = date_trade.strftime("%Y-%m-%d")
                # Running the model
                if "FCR-D" in self.markets:
                    if self.quality:
                        obj_ra, zeta_ra, nu_ra, P_DA_ra, P_act_ra, c_up_ra, c_dn_ra, SOC_ra = self.cross_validate_STC_FCRD_DA_IMB(date_trade)
                    else:
                        obj_ra, zeta_ra, nu_ra, P_DA_ra, P_act_ra, c_up_ra, c_dn_ra, SOC_ra = self.STC_FCRD_DA_IMB_daily(date_trade, self.W, self.lambda_Stc_FCR_D_up, self.lambda_Stc_FCR_D_dn, self.lambda_Stc_DA, self.lambda_Stc_im, self.r_Stc_up, self.r_Stc_dn)
                    c_up_arr_ra = np.append(c_up_arr_ra, c_up_ra)
                    c_dn_arr_ra = np.append(c_dn_arr_ra, c_dn_ra)
                elif self.markets == "DA with imbalance":
                    if self.quality:
                        obj_ra, zeta_ra, nu_ra, P_DA_ra, P_act_ra, SOC_ra = self.cross_validate_STC_FCRD_DA_IMB(date_trade)
                    else:
                        obj_ra, zeta_ra, nu_ra, P_DA_ra, P_act_ra, SOC_ra = self.STC_FCRD_DA_IMB_daily(date_trade, self.W, self.lambda_Stc_FCR_D_up, self.lambda_Stc_FCR_D_dn, self.lambda_Stc_DA, self.lambda_Stc_im, self.r_Stc_up, self.r_Stc_dn)
                zeta_arr_ra =  np.append(zeta_arr_ra, zeta_ra)
                nu_arr_ra = np.hstack((nu_arr_ra, nu_ra.reshape(1,self.W)))
                P_DA_arr_ra =  np.append(P_DA_arr_ra, P_DA_ra)
                P_act_arr_ra = np.hstack((P_act_arr_ra, P_act_ra))
                SOC_arr_ra = np.hstack((SOC_arr_ra, SOC_ra))

            #print("zeta:",zeta_arr_ra)
            #print("nu_arr_ra:",nu_arr_ra)
            #print("P_DA_arr_ra:",P_DA_arr_ra)
            #print("P_act_arr_ra:",P_act_arr_ra)
            # Generate individual revenues assuming equiprobable scenarios
            if "FCR-D" in self.markets:
                rev_FCR_D_ra[i] = (self.lambda_Stc_FCR_D_up_full.mean(axis=0) * c_up_arr_ra + self.lambda_Stc_FCR_D_dn_full.mean(axis=0) * c_dn_arr_ra).sum()
            else:
                rev_FCR_D_ra[i] = (np.zeros(len(P_DA_arr_ra))).sum()
            rev_DA_ra[i] = (self.lambda_Stc_DA_full.mean(axis=0) * (P_DA_arr_ra / self.n_DA)).sum()
            rev_IM_ra[i] = (self.lambda_Stc_im_full * ((P_act_arr_ra - P_DA_arr_ra) / self.n_im)).mean(axis=0).sum()
            #print("rev_FCR_D_ra[i]:",rev_FCR_D_ra[i])
            #print("rev_DA_ra[i]:",rev_DA_ra[i])
            self.PnL_ra[i] = rev_FCR_D_ra[i] + rev_DA_ra[i] + rev_IM_ra[i]

            self.CVaR_arr[i] = ( zeta_arr_ra - (1/(1-self.alpha_risk)) * (nu_arr_ra.mean(axis=1)) ).sum()
        
        # Reset to original
        self.beta_risk = beta_risk_original

        # Clip extreme -ve values
        np.clip(self.CVaR_arr,a_min=0,a_max=None)
        
        return #PnL_ra, CVaR_arr

    def plot_risk(self):
        """
        Scatter plot of Expected Profit vs CVaR (efficiency frontier),
        color-coded by beta in a single-hue gradient from 0 to 1.
        Excludes beta = 0 points (pathological CVaR).
        """

        # Run the risk analysis function
        self.risk_analysis()

        beta = np.asarray(self.beta_arr)
        pnl  = np.asarray(self.PnL_ra)
        cvar = np.asarray(self.CVaR_arr)

        # --- Basic sanity checks ---
        assert beta.shape == pnl.shape == cvar.shape, \
            "beta_arr, PnL_ra, and CVaR_arr must have the same shape"

        # --- Mask out beta = 0 (pathological CVaR ≈ -1e20) ---
        mask = beta > 0

        beta = beta[mask]
        pnl  = pnl[mask]
        cvar = cvar[mask]

        # --- Create figure ---
        plt.figure(figsize=(7, 5))

        # Scatter with single-hue gradient
        sc = plt.scatter(
            cvar,
            pnl,
            c=beta,
            cmap="Blues",        # single-hue colormap
            s=60,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9
        )

        # Connect points to emphasize the frontier shape
        plt.plot(cvar, pnl, color="gray", alpha=0.4, linewidth=1.2)

        # Colorbar for beta
        cbar = plt.colorbar(sc)
        cbar.set_label("Tuning parameter β", fontsize=11)

        # Labels and title
        plt.xlabel("CVaR (Risk)", fontsize=11)
        plt.ylabel("Expected Profit (PnL)", fontsize=11)
        plt.title("Risk–Return Efficiency Frontier (β > 0)", fontsize=12)

        # Grid + layout
        plt.grid(True, alpha=0.3)
        # Show plot
        plt.tight_layout()
        plt.savefig("risk_result_illustration_example", dpi=500)
        return plt.gcf()


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