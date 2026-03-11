# Import packages
from src import BESS_trading_strategy, API_KEY_ENTSOE
from datetime import date
import panel as pn
import panel.widgets as pnw
from entsoe import EntsoePandasClient

# Initialize the request protocol
CLIENT = EntsoePandasClient(api_key=API_KEY_ENTSOE)

# Trading Dashboard
#pn.extension('tabulator', 'katex')
# pn.extension('bokeh')

# ---------------------------------------------------------------------
# --- Helper: initialization status and result messages
# ---------------------------------------------------------------------
status_text = pn.pane.Markdown("#### ⚙️ Waiting for input...")
solve_status = pn.pane.Markdown("")
revenue_text = pn.pane.Markdown("")
mode_status = pn.pane.Markdown("")
plot_pane_main = pn.pane.Matplotlib(sizing_mode="stretch_width")
plot_pane_risk = pn.pane.Matplotlib(sizing_mode="stretch_width")
plot_pane_risk.visible = False
plot_button = None

# ---------------------------------------------------------------------
# --- 1. Top Row — General inputs
# ---------------------------------------------------------------------
country_display = pn.pane.Markdown("**Country:** 🇫🇷 France")

start_date_picker = pnw.DatePicker(
    name="Start Date", value=date(2025, 12, 25),
    start=date(2024, 1, 1), end=date(2026, 12, 31)
)
end_date_picker = pnw.DatePicker(
    name="End Date", value=date(2025, 12, 25),
    start=date(2024, 1, 1), end=date(2026, 12, 31)
)

row1 = pn.Row(country_display, start_date_picker, end_date_picker)

# ---------------------------------------------------------------------
# --- 2. Markets and Scheme selection
# ---------------------------------------------------------------------
markets_button = pnw.RadioButtonGroup(
    name="Markets", value="DA with imbalance",
    options=["FCR-D + DA with imbalance", "DA with imbalance"]
)

scheme_button = pnw.RadioButtonGroup(
    name="Scheme", value="Deterministic",
    options=["Deterministic", "Stochastic"]
)

# If stochastic selected → show scenario slider
scenario_slider = pnw.IntSlider(
    name="Number of Scenarios (W)", start=5, end=100, value=10, visible=False
)

# --- Advanced stochastic options (hidden by default) ---
solution_mode = pnw.Select(
    name="Solution Mode",
    options=["Standard", "Ex-post Quality", "Risk Analysis"],
    value="Standard",
    visible=False
)

alpha_slider = pnw.IntSlider(
    name="CVaR Confidence Level α (%)",
    start=0, end=100, step=10, value=90, visible=False
)

beta_slider = pnw.FloatSlider(
    name="Risk Aversion β",
    start=0.0, end=1.0, step=0.1, value=0.5, visible=False
)

row2b = pn.Row(solution_mode, alpha_slider, beta_slider)
row2b.visible = False

@pn.depends(scheme_button, watch=True)
def update_scenario_visibility(scheme):
    is_stoch = (scheme == "Stochastic")
    scenario_slider.visible = is_stoch
    solution_mode.visible = is_stoch
    row2b.visible = is_stoch

    # Reset advanced options when leaving stochastic
    if not is_stoch:
        solution_mode.value = "Standard"
        alpha_slider.visible = False
        beta_slider.visible = False

        plot_pane_risk.visible = False
        plot_pane_risk.object = None

        # Reset solve flags
        global solve_quality, solve_risk, alpha_risk, beta_risk
        solve_quality = False
        solve_risk = False
        alpha_risk = 1
        beta_risk = 0


@pn.depends(solution_mode, watch=True)
def update_risk_controls(mode):
    if mode == "Risk Analysis":
        alpha_slider.visible = True
        beta_slider.visible = True
    else:
        alpha_slider.visible = False
        beta_slider.visible = False

@pn.depends(solution_mode, alpha_slider, beta_slider, watch=True)
def update_solve_flags(mode, alpha, beta):
    global solve_quality, solve_risk, alpha_risk, beta_risk

    # Reset defaults
    solve_quality = False
    solve_risk = False
    alpha_risk = 1
    beta_risk = 0

    if mode == "Ex-post Quality":
        solve_quality = True

    elif mode == "Risk Analysis":
        solve_risk = True
        alpha_risk = alpha / 100.0   # convert % → [0,1]
        beta_risk = beta
    
    plot_pane_risk.visible = False
    plot_pane_risk.object = None

row2 = pn.Row(markets_button, scheme_button, scenario_slider)

# ---------------------------------------------------------------------
# --- 3. Asset parameters
# ---------------------------------------------------------------------
asset_title = pn.pane.Markdown("### ⚡ Asset Parameters")

SOC_max_input = pnw.FloatInput(name="Storage Capacity (MWh)", value=20.0, step=0.1)
R_max_input = pnw.FloatInput(name="Power Capacity (MW)", value=10.0, step=0.1)

row3 = pn.Row(SOC_max_input, R_max_input)

# ---------------------------------------------------------------------
# --- 4. Additional parameters (hidden by default)
# ---------------------------------------------------------------------
additional_params_expander = pn.widgets.Toggle(name="Additional Parameters", value=False)

FCR_max_input = pnw.FloatInput(name="Maximum allowable FCR bid (% of storage capacity)", value=80.0, step=1.0)
SOC_i_input = pnw.FloatInput(name="Initial State-of-Charge (MWh)", value=0.0, step=0.1)

eta_c_display = pn.pane.Markdown("η_charge = 100%")
eta_d_display = pn.pane.Markdown("η_discharge = 100%")
nu_deg_display = pn.pane.Markdown("ν_degradation = 0 EUR/MWh")

row4 = pn.Row(FCR_max_input, SOC_i_input, eta_c_display, eta_d_display, nu_deg_display)
row4.visible = False

@pn.depends(additional_params_expander, watch=True)
def toggle_additional_params(toggle):
    row4.visible = toggle

# ---------------------------------------------------------------------
# --- 5. Load Button
# ---------------------------------------------------------------------
load_button = pnw.Button(name="Load Data", button_type="primary")

trade = None  # placeholder for the BESS_trading_strategy object
# --- Internal flags for solve options ---
solve_quality = False
solve_risk = False
alpha_risk = 1
beta_risk = 0

def load_data(event):
    global trade
    start = start_date_picker.value
    end = end_date_picker.value
    #print(start)

    if end < start:
        status_text.object = "⚠️ **End date must be after start date. Please adjust.**"
        return

    status_text.object = "⏳ Loading data..."

    try:
        # Create instance
        trade = BESS_trading_strategy(
            CLIENT=CLIENT,
            SOC_max=SOC_max_input.value,
            R_max=R_max_input.value,
            SOC_i=SOC_i_input.value,
            FCR_max=FCR_max_input.value / 100,
            scheme=scheme_button.value,
            W=scenario_slider.value,
            markets=markets_button.value,
            country="FR",
            start_date=str(start),
            end_date=str(end),
        )
        status_text.object = "✅ Data loaded successfully! Ready to solve."
        show_solve_section()
    except Exception as e:
        status_text.object = f"❌ Error while loading: `{e}`"


# ---------------------------------------------------------------------
# --- 6. Solve Button (appears after Load)
# ---------------------------------------------------------------------
solve_button = pnw.Button(name="Solve Optimization", button_type="success")
plot_button = pnw.Button(name="Plot Results", button_type="primary", visible=False)

def show_solve_section():
    solve_button.visible = True

def run_solve(event):
    if trade is None:
        solve_status.object = "⚠️ Please load data first."
        return

    solve_status.object = "🧩 Solving optimization..."
    try:
        trade.solve(
            quality=solve_quality,
            risk=solve_risk,
            alpha_risk=alpha_risk,
            beta_risk=beta_risk
        )
        solve_status.object = "✅ Optimization complete."
        #rev = trade.revenue()
        revenue_text.object = trade.revenue()
        plot_button.visible = True
    except Exception as e:
        solve_status.object = f"❌ Error during solve: `{e}`"

def run_plot(event):
    if trade is not None:
        fig = trade.plot()
        plot_pane_main.object = fig

        # Only plot risk results if Risk Analysis was selected
        if solve_risk:
            fig_risk = trade.plot_risk()
            plot_pane_risk.object = fig_risk
            plot_pane_risk.visible = True
        else:
            plot_pane_risk.object = None
            plot_pane_risk.visible = False

    else:
        solve_status.object = "⚠️ Load and solve before plotting."

load_button.on_click(load_data)
solve_button.on_click(run_solve)
plot_button.on_click(run_plot)

# Initially hidden
solve_button.visible = False
plot_button.visible = False

# ---------------------------------------------------------------------
# --- Combine all panels
# ---------------------------------------------------------------------
dashboard = pn.Column(
    pn.pane.Markdown("## ⚙️ BESS Trading Strategy - Interactive Dashboard"),
    row1,
    row2,
    row2b,
    asset_title,
    row3,
    additional_params_expander,
    row4,
    pn.Row(load_button),
    status_text,
    pn.Row(solve_button),
    solve_status,
    revenue_text,
    plot_button,
    pn.Column(plot_pane_main, plot_pane_risk),
)

dashboard.servable()

# If running as standalone script, open in browser
if __name__ == "__main__":
    pn.serve(dashboard)  # ✅ use your actual layout variable name
