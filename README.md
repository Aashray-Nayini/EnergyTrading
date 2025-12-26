Overview of the package
---------------------------------
EnergyTrading package is aimed at providing multi-market power trading strategies for Renewable Energy and BESS assets. Take a look at the dependencies section before running the model. While execution, the code starts a Panel web server, opens a browser tab, and serves your dashboard at a local URL (default http://localhost:5006).

Dependencies
---------------------------------
1. A list of libraries needed to run the code is saved in requirements.txt file. Follow the instructions from the section below to install them on your virtual environment
2. An api key to access data from entso-e transparency platform. Following the steps in the [official documentation](https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html?ref=datons.ai#_authentication_and_authorisation), you can register on the [ENTSO-E page](https://transparency.entsoe.eu/?ref=datons.ai), and then you need to send an email to transparency@entsoe.eu with the subject “Restful API access”. After procuring your key, save it in a .env file in this package with a name "API_KEY".


Virtual Environment and Installations
---------------------------------

It is recommended to make a virtual environment and install the required packages before running the package.

1. On the target machine, create a virtual environment:

    `python3 -m venv [name of the environment]`

2. Activate the environment (replace "myenv" with the name of your virtual environment):

    `source myenv/bin/activate`

3. Install the required packages from the `requirements.txt` file (make sure to give the full path of this file):

    `pip install -r path/to/requirements.txt`

Interactive Trading Dashboard
---------------------------------
By running the `BESS.py` file, an interactive dashboard will be shown to optimize and trade power from a BESS asset. The dashboard allows users to configure asset parameters, select market participation schemes, solve the optimization problem, and visualize results.

- What you can do in the dashboard
    - Select country (set to France for now) and time horizon
    - Choose market participation
    - Select optimization scheme (eg. stochastic with 100 scenarios)
    - Configure asset parameters
    - Run and analyze results

The dashboard runs in a browser using Panel + Matplotlib and updates dynamically based on user inputs. Here is a snapshot of how the dashboard looks:

<p align="center">
<img src="docs/images/trading_desk.png" width="700"/>
</p>

Trading Strategy
---------------------------------
A full documentation of the mathematical formulation can be found [here](docs/formulation.md).

The objective funtion dealing with FCR-D and Day-Ahead with imbalance settlement is given by:

<p align="center">
<img src="docs/images/Obj_FCRD_DA_IM.png" width="500"/>
</p>

Result Illustration
---------------------------------

The output from the trading algorithms produces 5 sub-plots:
1. Price forecasts used in the algorithm, including the uncertainity range,
2. Bids to be submitted in the Day-Ahead Market,
3. Bids to be submitted in the Anscillary service market Market,
4. State of charge evolution during the planning period assuming the submitted bids are accepted. The plots also includes the uncertainity range as the balancing activation is not deterministic,
5. The Profit & Loss evolution in different markets and of overall strategy.

<p align="center">
<img src="docs/images/result_illustration_example.png" width="1000"/>
</p>

## 👤 Author

Nayini Venkat Aashray [[LinkedIn](https://www.linkedin.com/in/aashraynayini/)]

Email: venkat-aashray.nayini@master.polytechnique.org 
