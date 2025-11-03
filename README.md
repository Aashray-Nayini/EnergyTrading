Overview of the package
---------------------------------
EnergyTrading package is aimed at providing trading strategies for Renewable Energy and BESS assets in different markets. Take a look at the dependencies section before running the model. While execution, the code starts a Panel web server, opens a browser tab, and serves your dashboard at a local URL (default http://localhost:5006).

Dependencies
---------------------------------
1. A list of libraries needed to run the code is saved in requirements.txt file. Follow the instructions from the section below to install them on your virtual environment
2. An api key to access data from entso-e transparancy paltform. Following the steps in the [official documentation](https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html?ref=datons.ai#_authentication_and_authorisation), you can register on the [ENTSO-E page](https://transparency.entsoe.eu/?ref=datons.ai), and then you need to send an email to transparency@entsoe.eu with the subject “Restful API access”. After procuring your key, save it in a .env file in this package with a name "API_KEY".


Virtual Environment and Installations
---------------------------------

It is recommended to make a virtual environment and install the required packages before running the package.

1. On the target machine, create a virtual environment:

    `python3 -m venv [name of the environment]`

2. Activate the environment (replace "myenv" with the name of your virtual environment):

    `source myenv/bin/activate`

3. Install the required packages from the `requirements.txt` file (make sure to give full path of this file):

    `pip install -r path/to/requirements.txt`

Mathematical Formulation
---------------------------------

### Sets

| Symbol | Description |
|:-------:|-------------|
| $ t $ | Index for time steps starting from 0 unpto T| 
| $ \omega $ | Index for stochastic scenarios out of W total scenarios |

---

### Parameters

| Symbol | Description | Units |
|:-------:|-------------|-----|
| $ \lambda_{t,\omega}^{FCR-D↑} $ | Activation price of FCR-D up regulation at time $ t $ and scenario $ \omega $ | €/MW |
| $ \lambda_{t,\omega}^{FCR-D↓} $ | Activation proce of FCR-D down regulation at time $ t $ and scenario $ \omega $ | €/MW |
| $ \lambda_{t,\omega}^{DA} $ | Day-ahead market price at time $ t $ and scenario $ \omega $ | €/MWh |
| $ \lambda_{t,\omega}^{↑} $ | Upward imbalance price at time $ t $ and scenario $ \omega $ | €/MWh |
| $ \lambda_{t,\omega}^{↓} $ | Downward imbalance price at time $ t $ and scenario $ \omega $ | €/MWh |
| $ M $ | Big-M parameter (large positive constant) | - |
| $ r_{t}^{↑} $ | Activation ratio of FCR-D up regulation capacity at time $ t $ | - |
| $ r_{t}^{↓} $ | Activation ratio of FCR-D down regulation capacity at time $ t $ | - |
| $ n $ | Hourly resolution: Number of time steps per hour (e.g., 4 for 15-min resolution) | - |
| $ q $ | Allowed violation probability (fraction of scenarios allowed to breach constraints) | %/100 |
| $ \overline{SOC} $ | Maximum state of charge | MWh |
| $ \overline{R} $ | Maximum power rating | MW |
| $ \overline{FCR} $ | Maximum FCR fraction of storage capacity | %/100 |
| $ {SOC}_{i} $ | Initial state of charge (MWh) | MWh |

---

### Decision Variables

| Symbol | Description | Units |
|:-------:|-------------|-----|
| $ P_{t}^{DA} $ | Day-ahead energy bid for the period $t$: **+ve for discharging** | MW |
| $ P_{t,\omega}^{act} $ | Actual dispatched power (MW) in scenario $ \omega $ at period $t$: **+ve for discharging**| MW |
| $ c_{t}^{↑} $ | FCR-D up regulation bid in the period $t$ (considered as accepted in full): **+ve for discharging** | MW |
| $ c_{t}^{↓} $ | FCR-D down regulation bid in the period $t$ (considered as accepted in full): **+ve for charging**| MW |
| $ {SOC}_{t,\omega} $ | State of charge (MWh) **at the end of period $ t $**, scenario $ \omega $ | MWh |
| $ I_{t,\omega}^{B} $ | Imbalance settlement income at period $ t $, scenario $ \omega $ | € |
| $ y_{t,\omega}^{↑} $ | Binary variable for FCD-D up violataion (1 for violation)| - |
| $ y_{t,\omega}^{↓} $ | Binary variable for FCD-D down violataion (1 for violation)| - |

---

### FCR-D-up + FCR-D-down + DA markets with imbalance settlement

$$
\max ~~~~~~ \dfrac{1}{\#\omega} \sum_{t} \sum_{\omega} 
\left(
\lambda_{t,\omega}^{FCR-D↑} r_{t}^{↑} c_{t}^{↑}
+ \lambda_{t,\omega}^{FCR-D↓} r_{t}^{↓} c_{t}^{↓}
+ \lambda_{t,\omega}^{DA} \dfrac{P_{t}^{DA}}{n}
+ I_{t,\omega}^{B}
\right)
$$

s.t.,  

$$
I_{t,\omega}^{B} =
\begin{cases}
\lambda_{t,\omega}^{↑}\dfrac{(P_{t,\omega}^{act}-P_{t}^{DA})}{n} & \text{if system is deficit}\\ \\
\lambda_{t,\omega}^{↓}\dfrac{(P_{t,\omega}^{act}-P_{t}^{DA})}{n} & \text{if system is surplus}
\end{cases}
\quad \forall t,\omega
$$

$$
\dfrac{r_{t}^{↑} c_{t}^{↑}}{n} \le {SOC}_{t-1,\omega} \quad \forall t,\omega
$$

$$
\dfrac{r_{t}^{↓} c_{t}^{↓}}{n} \le \overline{SOC} - {SOC}_{t-1,\omega} \quad \forall t,\omega
$$

$$
\dfrac{c_{t}^{↑} + c_{t}^{↓}}{n} \le \overline{FCR} \cdot \overline{SOC} \quad \forall t
$$

$$
P_{t,\omega}^{act} - \overline{R} + r_{t}^{↑} c_{t}^{↑} \le y_{t,\omega}^{↑}M
$$

$$
P_{t,\omega}^{act} + \overline{R} - r_{t}^{↓} c_{t}^{↓} \ge y_{t,\omega}^{↓}M
$$

$$
\sum_{\omega} y_{t,\omega}^{↑} \le qW \quad \forall t
$$

$$
\sum_{\omega} y_{t,\omega}^{↓} \le qW \quad \forall t
$$

$$
{SOC}_{0,\omega} = {SOC}_{i} - \dfrac{P_{0,\omega}^{act}}{n} \quad \forall \omega
$$

$$
{SOC}_{t,\omega} = {SOC}_{t-1,\omega} - \dfrac{P_{t,\omega}^{act}}{n} \quad \forall t > 0, \omega
$$

$$
0 \le {SOC}_{t,\omega} \le \overline{SOC} \quad \forall t, \omega
$$

$$
-\overline R \le P_{t}^{DA} \le \overline R \quad \forall t
$$

$$
-\overline R \le P_{t,\omega}^{act} \le \overline R \quad \forall t
$$

$$
c_{t}^{↑}, c_{t}^{↓} \ge 0 \quad \forall t
$$

$$
y_{t,\omega}^{↑}, y_{t,\omega}^{↓} \in \{0,1\} \quad \forall t,\omega
$$

For France and Germany:

$$
c_{t}^{↑} = c_{t'}^{↑} \quad \forall t = \{0,\tfrac{T}{6},\tfrac{2T}{6},\tfrac{3T}{6},\tfrac{4T}{6},\tfrac{5T}{6}\}, ~ t' = \{t+1,t+2,...,t+\tfrac{T}{6}-1\}
$$

$$
c_{t}^{↓} = c_{t'}^{↓} \quad \forall t = \{0,\tfrac{T}{6},\tfrac{2T}{6},\tfrac{3T}{6},\tfrac{4T}{6},\tfrac{5T}{6}\}, ~ t' = \{t+1,t+2,...,t+\tfrac{T}{6}-1\}
$$

### Further Information and References

- Rolling intrinsic algorithm for continuous intraday: Leo Semmelmann, Jannik Dresselhaus, Kim K. Miskiw, Jan Ludwig, and Christof Weinhardt. 2025. An Algorithm for Modelling Rolling Intrinsic Battery Trading on the Continuous Intraday Market. SIGENERGY Energy Inform. Rev. 4, 4 (October 2024), 163–174. https://doi.org/10.1145/3717413.3717428 
- On top of the above formulation, added efficiencies, degradation costs and frequency results: Schaurecker, D., Wozabal, D., Löhndorf, N. and Staake, T., 2025. Maximizing Battery Storage Profits via High-Frequency Intraday Trading. arXiv preprint arXiv:2504.06932.
- FCR + Continuous Intraday: Zhang, Y., Ridinger, W. and Wozabal, D., 2025. Joint Bidding on Intraday and Frequency Containment Reserve Markets. arXiv preprint arXiv:2510.03209.
- Day Ahead + Continuous Intraday: Oeltz, D. and Pfingsten, T., 2025. Rolling intrinsic for battery valuation in day-ahead and intraday markets. arXiv preprint arXiv:2510.01956.
- "Imbalance and open-loop ACE show no correlation with wind power, PV generation, or consumption forecasts, as well as the actual realizations of these variables": Dumas, J., Finet, S., Grisey, N., Hamdane, I. and Plessiez, P., 2025. Analysis of the French system imbalance paving the way for a novel operating reserve sizing approach. arXiv preprint arXiv:2503.24240.


## 👤 Author

Nayini Venkat Aashray [LinkedIn](https://www.linkedin.com/in/aashraynayini/)

Email: venkat-aashray.nayini@master.polytechnique.org 
