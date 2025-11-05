#%%
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import njit
import plotly.graph_objects as go
from scipy.interpolate import griddata

st.set_page_config(page_title="Accumulator/Decumulator Pricing Dashboard", layout="wide")

st.markdown("""
    <style>
    .block-container {
        max-width: 1600px;  /* wider but still limited */
        margin: 0 auto;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


@njit()
def CND(x):
    """
    Cumulative normal distribution using Abramowitz and Stegun approximation,
    (negligibile error if compared to scipy)
    Needed as numba doesnt support scipy.

    Args:
        x (float or np.array): normal random variable [Nx1]

    Returns:
        float or np.array: [Nx1] np array of CNDs
    """    
    a1, a2, a3, a4, a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    gamma = 0.2316419
    abs_x = np.abs(x)
    t = 1 / (1 + gamma * abs_x)
    poly = (a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5)
    exp_term = np.exp(-0.5 * abs_x**2) / np.sqrt(2 * np.pi)
    result = 1 - exp_term * poly
    enconder_a = np.where(x < 0, 1, 0)
    enconder_b = np.where(x < 0, -1, 1)

    return enconder_a + enconder_b*result 

@njit
def StandardBarrierOption(TypeFlag, S, X, H, K, Time, r, b, sigma, Fwd_Time):

    # References:
    #   Haug, Chapter 2.10.1

    '''
    
    Typeflag is an integer build of three digits mapped according to the following order:

    a) call / put  = 1 / 2
    b) up   / down = 1 / 2
    c) in   / out  = 1 / 2 
    
    ---> TypeFlag = abc

    cui = 111 ---> up-and-in call
    cuo = 112 ---> up-and-out call
    cdi = 121 ---> down-and-in call
    cdo = 122 ---> down-and-out call

    pui = 211 ---> up-and-in put
    puo = 212 ---> up-and-out put
    pdi = 221 ---> down-and-in put
    pdo = 222 ---> down-and-out put

    '''

    mu = (b - sigma ** 2 / 2) / sigma ** 2
    lambda_ = np.sqrt(mu ** 2 + 2 * r / sigma ** 2)

    d1 = (np.log(S / X) + (b + 0.5 * sigma ** 2) * Time) / (sigma * np.sqrt(Time))
    d2 = d1 - (sigma * np.sqrt(Time))

    X1 = np.log(S / X) / (sigma * np.sqrt(Time)) + (1 + mu) * sigma * np.sqrt(Time)
    X2 = np.log(S / H) / (sigma * np.sqrt(Time)) + (1 + mu) * sigma * np.sqrt(Time)
    y1 = np.log(H ** 2 / (S * X)) / (sigma * np.sqrt(Time)) + (1 + mu) * sigma * np.sqrt(Time)
    y2 = np.log(H / S) / (sigma * np.sqrt(Time)) + (1 + mu) * sigma * np.sqrt(Time)
    Z = np.log(H / S) / (sigma * np.sqrt(Time)) + lambda_ * sigma * np.sqrt(Time)

    # Determine eta and phi based on TypeFlag
    if TypeFlag in [121, 122]:
        eta, phi = 1, 1
    elif TypeFlag in [111, 112]:
        eta, phi = -1, 1
    elif TypeFlag in [221, 222]:
        eta, phi = 1, -1
    elif TypeFlag in [211, 212]:
        eta, phi = -1, -1

    # Calculate terms f1 to f6
    f1 = (phi * S * np.exp((b - r) * Fwd_Time) * CND(phi * X1) -
    phi * X * np.exp(-r * Fwd_Time) * CND(phi * X1 - phi * sigma * np.sqrt(Time)))
    f2 = (phi * S * np.exp((b - r) * Fwd_Time) * CND(phi * X2) -
    phi * X * np.exp(-r * Fwd_Time) * CND(phi * X2 - phi * sigma * np.sqrt(Time)))
    f3 = (phi * S * np.exp((b - r) * Fwd_Time) * (H / S) ** (2 * (mu + 1)) *
    CND(eta * y1) - phi * X * np.exp(-r * Fwd_Time) * (H / S) ** (2 * mu) *
    CND(eta * y1 - eta * sigma * np.sqrt(Time)))
    f4 = (phi * S * np.exp((b - r) * Fwd_Time) * (H / S) ** (2 * (mu + 1)) *
    CND(eta * y2) - phi * X * np.exp(-r * Fwd_Time) * (H / S) ** (2 * mu) *
    CND(eta * y2 - eta * sigma * np.sqrt(Time)))
    f5 = (K * np.exp(-r * Fwd_Time) * (CND(eta * X2 - eta * sigma *
    np.sqrt(Time)) - (H / S) ** (2 * mu) * CND(eta * y2 - eta *
    sigma * np.sqrt(Time))))
    f6 = (K * ((H / S) ** (mu + lambda_) * CND(eta * Z) + (H / S)**(mu - lambda_) *
    CND(eta * Z - 2 * eta * lambda_ * sigma * np.sqrt(Time))))

    # Determine StandardBarrier based on TypeFlag and conditions
    # -------------------------------------------------------------------------------------------

    if TypeFlag == 112:
        # if S >= H:
        #     BarrierOption = 0 # expires worthless
        # else: 
        #     if X >= H:
        #         BarrierOption = f6
        #     else:
        #         BarrierOption = f1 - f2 + f3 - f4 + f6

                
        BarrierOption = np.where(
            S >= H,
            0, # expires worthless
            np.where(X >= H, f6, f1 - f2 + f3 - f4 + f6)
        )

    # -------------------------------------------------------------------------------------------

    elif TypeFlag == 111:
        # if S >= H: # simple vanilla call
        #     BarrierOption = S * np.exp((b-r)*Fwd_Time) * CND(d1) - X * np.exp((-r)*Fwd_Time) * CND(d2)
        # else: 
        #     if X >= H:
        #         BarrierOption = f1 + f5
        #     else:
        #         BarrierOption = f2 - f3 + f4 + f5

        BarrierOption = np.where(
            S >= H,
            S * np.exp((b-r)*Fwd_Time) * CND(d1) - X * np.exp((-r)*Fwd_Time) * CND(d2), # standard vanilla call
            np.where(X >= H, f1 + f5, f2 - f3 + f4 + f5)
        )

    # -------------------------------------------------------------------------------------------
    
    elif TypeFlag == 122:
        # if S >= H:
        #     if X >= H:
        #         BarrierOption = f1 - f3 + f6
        #     else:
        #         BarrierOption = f2 + f6 - f4
        # else:
        #     BarrierOption = 0 # expires worthless
    
        BarrierOption = np.where(
            S >= H,
            np.where(X >= H,  f1 - f3 + f6, f2 + f6 - f4),
           0 # expires worthless
        )

    # -------------------------------------------------------------------------------------------
    
    elif TypeFlag == 121:
        # if S >= H:
        #     if X >= H:
        #         BarrierOption = f3 + f5
        #     else:
        #         BarrierOption = f1 - f2 + f4 + f5
        # else: # simple vanilla call
        #     BarrierOption = S * np.exp((b-r)*Fwd_Time) * CND(d1) - X * np.exp((-r)*Fwd_Time) * CND(d2)

        BarrierOption = np.where(
            S >= H,
            np.where(X >= H, f3 + f5, f1 - f2 + f4 + f5),
           S * np.exp((b-r)*Fwd_Time) * CND(d1) - X * np.exp((-r)*Fwd_Time) * CND(d2) # simple vanilla call
        )


    # -------------------------------------------------------------------------------------------

    elif TypeFlag == 222:
        # if S >= H:
        #     if X >= H:
        #         BarrierOption = f1 - f2 + f3 - f4 + f6
        #     else:
        #         BarrierOption = f6
        # else:
        #     BarrierOption = 0 # expires worthless

        BarrierOption = np.where(
            S >= H,
            np.where(X >= H, f1 - f2 + f3 - f4 + f6, f6),
            0 # expires worthless
        )

    # -------------------------------------------------------------------------------------------

    elif TypeFlag == 221:

        BarrierOption = np.where(
            S >= H,
            np.where(X >= H, f2 - f3 + f4 + f5, f1 + f5),
            X * np.exp((-r)*Fwd_Time) * CND(-d2) - S * np.exp((b-r)*Fwd_Time) * CND(-d1) # simple vanilla put
        )

    # -------------------------------------------------------------------------------------------

    elif TypeFlag == 212:

        BarrierOption = np.where(
            S >= H,
            0, # expires worthless
            np.where(X >= H, f2 - f4 + f6, f1 - f3 + f6)
        )

    # -------------------------------------------------------------------------------------------

    elif TypeFlag == 211:

        BarrierOption = np.where(
            S >= H,
            (X * np.exp(-r * Fwd_Time) * CND(-d2) - S * np.exp((b - r) * Fwd_Time) * CND(-d1)), # simple vanilla put
            np.where(X >= H, f1 - f2 + f4 + f5, f3 + f5)
        )


    # -------------------------------------------------------------------------------------------
    
    return np.maximum(BarrierOption, 0)
mpl.rcParams['figure.dpi'] = 300


# -------------------------

# --- COLUMNS ---
col1, spacer, col2 = st.columns([1.1, 0.2, 2])  # small gap between them

# --- LEFT PANEL: Inputs ---
with col1:
    st.header("⚙️ Model Parameters")

    colA, colB = st.columns(2)

    with colA:
        type_ = st.selectbox("Product Type", ["Accumulator", "Decumulator"])
        strike = st.number_input("Strike %", value=90.0 if type_ == "Accumulator" else 110.0)
        barrier = st.number_input("Barrier %", value=110.0 if type_ == "Accumulator" else 90.0)
        gear = st.number_input("Gear", 0, 2, 2)
        nominal = st.number_input("Nominal", value=10000.0)
        
        periods = st.slider("Number of Periods", 1, 52, 52)
        periods_guaranteed = st.slider("Guaranteed Periods", 0, 10, 0)
        if periods_guaranteed > periods:
            st.error("Guaranteed periods cannot exceed total periods.")
            periods_guaranteed = periods

    with colB:
        # s0 = st.number_input("Spot Price", value=100.0)
        rate = st.number_input("Domestic Interest Rate", value=0.03)
        div = st.number_input("Foreign Interest Rate/Dividend Yield", value=0.0)
        vol = st.number_input("Volatility", value=0.1, min_value=0.01, max_value=1.0, step=0.01)
        shock = st.number_input("Margin Risk Factor", 0.01, 0.2, 0.1)
        
        

# --- CSS for full-height right column ---
st.markdown("""
<style>
[data-testid="column"]:nth-of-type(2) {
    height: 120vh;
    display: flex;
    flex-direction: column;
}
.full-height-container {
    display: flex;
    flex-direction: column;
    flex: 1;
    justify-content: space-between;
}
.full-height-container > div {
    flex: 1;
}
</style>
""", unsafe_allow_html=True)

# --- RIGHT PANEL ---
with col2:
    st.header("📈 Charts")
    st.markdown('<div class="full-height">', unsafe_allow_html=True)
    container_2d = st.container()
    container_3d = st.container()

    # --- GRID & calculations ---
    if type_ == "Accumulator":
        spots_ = np.arange(0.6 * strike, 1.2 * barrier, 1)
        around_b = np.arange(1.1 * strike, 1.01 * barrier, 0.1)
        spots_ = np.unique(np.concatenate([spots_, around_b]))
    else:
        spots_ = np.arange(0.6 * barrier, 1.2 * strike, 1)
        around_b = np.arange(0.99 * barrier, 1.01*strike, 0.1)
        spots_ = np.unique(np.concatenate([spots_, around_b]))

    spots_plus = spots_ * (1+shock)
    spots_minus = spots_ * (1-shock)
    spots = np.unique(np.concatenate([spots_, spots_plus, spots_minus]))

    strips_mat = np.arange(1, periods + 1) / periods
    guaranteed_mats = strips_mat[:periods_guaranteed]
    sign_call, sign_put = (1, -1) if type_ == "Accumulator" else (-1, 1)
    gear_call, gear_put = (1, gear) if type_ == "Accumulator" else (gear, 1)
    call_type = 112 if type_ == "Accumulator" else 122
    put_type = 212 if type_ == "Accumulator" else 222
    h = 1e-6

    # This # empty structures 

    call_price   = []
    put_price    = []
    strip_price  = []

    delta_call   = []
    delta_put    = []
    delta_strip  = []

    call_price_   = []
    put_price_    = []
    strip_price_  = []

    delta_call_   = []
    delta_put_    = []
    delta_strip_  = []

    # call_price_van  = []
    # put_price_van   = []
    # call_price_van_ = []
    # put_price_van_  = []

    # delta_call_van  = []
    # delta_put_van   = []
    # delta_call_van_ = []
    # delta_put_van_  = []

    # fwd_price = []
    # fwd_price_ = []

    for spot in spots:
        for t in strips_mat:
            maturity = t
            nominal_ = gear * nominal / periods
            # fw = sign_call*(spot - strike * np.exp(-rate * maturity))

            if t in guaranteed_mats:
                c_price = StandardBarrierOption(TypeFlag=111, S=spot, X=strike, H=1e-5, K=0, Time=maturity, r=rate, b=rate - div, sigma=vol, Fwd_Time=maturity)
                p_price = StandardBarrierOption(TypeFlag=211, S=spot, X=strike, H=1e-5, K=0, Time=maturity, r=rate, b=rate - div, sigma=vol, Fwd_Time=maturity)

                c_price_up = StandardBarrierOption(TypeFlag=111, S=spot + h, X=strike, H=1e-5, K=0, Time=maturity, r=rate, b=rate - div, sigma=vol, Fwd_Time=maturity)
                c_price_do = StandardBarrierOption(TypeFlag=111, S=spot - h, X=strike, H=1e-5, K=0, Time=maturity, r=rate, b=rate - div, sigma=vol, Fwd_Time=maturity)

                p_price_up = StandardBarrierOption(TypeFlag=211, S=spot + h, X=strike, H=1e-5, K=0, Time=maturity, r=rate, b=rate - div, sigma=vol, Fwd_Time=maturity)
                p_price_do = StandardBarrierOption(TypeFlag=211, S=spot - h, X=strike, H=1e-5, K=0, Time=maturity, r=rate, b=rate - div, sigma=vol, Fwd_Time=maturity)

                call_price.append(sign_call * c_price * nominal_ )
                put_price.append(sign_put  * p_price * nominal_ )
                strip_price.append((sign_call * c_price + sign_put * p_price )* nominal_)

                d_call = (c_price_up - c_price_do) / (2*h)
                d_put = (p_price_up - p_price_do) / (2*h)
                delta_call .append (sign_call * d_call * nominal_)
                delta_put  .append (sign_put    *d_put * nominal_)
                delta_strip.append ((sign_call  * d_call + sign_put   * d_put) * nominal_)

            else:
                c_price = StandardBarrierOption(TypeFlag=call_type, S=spot, X=strike, H=barrier, K=0, Time=maturity, r=rate, b=rate - div, sigma=vol, Fwd_Time=maturity)
                p_price = StandardBarrierOption(TypeFlag=put_type, S=spot, X=strike, H=barrier, K=0, Time=maturity, r=rate, b=rate - div, sigma=vol, Fwd_Time=maturity)

                c_price_up = StandardBarrierOption(TypeFlag=call_type, S=spot + h, X=strike, H=barrier, K=0, Time=maturity, r=rate, b=rate - div, sigma=vol, Fwd_Time=maturity)
                c_price_do = StandardBarrierOption(TypeFlag=call_type, S=spot - h, X=strike, H=barrier, K=0, Time=maturity, r=rate, b=rate - div, sigma=vol, Fwd_Time=maturity)

                p_price_up = StandardBarrierOption(TypeFlag=put_type, S=spot + h, X=strike, H=barrier, K=0, Time=maturity, r=rate, b=rate - div, sigma=vol, Fwd_Time=maturity)
                p_price_do = StandardBarrierOption(TypeFlag=put_type, S=spot - h, X=strike, H=barrier, K=0, Time=maturity, r=rate, b=rate - div, sigma=vol, Fwd_Time=maturity)
                
                call_price.append(sign_call * c_price * nominal_ * gear_call)
                put_price.append(sign_put  * p_price * nominal_ * gear_put)
                strip_price.append((sign_call * c_price * gear_call + sign_put * p_price * gear_put)* nominal_)

                d_call = (c_price_up - c_price_do) / (2*h)
                d_put = (p_price_up - p_price_do) / (2*h)
                delta_call .append (sign_call * gear_call* d_call * nominal_)
                delta_put  .append (sign_put  * gear_put  *d_put * nominal_)
                delta_strip.append ((sign_call * gear_call * d_call + sign_put  * gear_put  * d_put) * nominal_)

            # vanilla instruments 
            # c_price_van = StandardBarrierOption(TypeFlag=111, S=spot, X=strike, H=1e-5, K=0, Time=maturity, r=rate, b=rate - div, sigma=vol, Fwd_Time=maturity)
            # p_price_van = StandardBarrierOption(TypeFlag=211, S=spot, X=strike, H=1e-5, K=0, Time=maturity, r=rate, b=rate - div, sigma=vol, Fwd_Time=maturity)

            # c_price_van_up = StandardBarrierOption(TypeFlag=111, S=spot+h, X=strike, H=1e-5, K=0, Time=maturity, r=rate, b=rate - div, sigma=vol, Fwd_Time=maturity)
            # p_price_van_up = StandardBarrierOption(TypeFlag=211, S=spot+h, X=strike, H=1e-5, K=0, Time=maturity, r=rate, b=rate - div, sigma=vol, Fwd_Time=maturity)

            # c_price_van_do = StandardBarrierOption(TypeFlag=111, S=spot-h, X=strike, H=1e-5, K=0, Time=maturity, r=rate, b=rate - div, sigma=vol, Fwd_Time=maturity)
            # p_price_van_do = StandardBarrierOption(TypeFlag=211, S=spot-h, X=strike, H=1e-5, K=0, Time=maturity, r=rate, b=rate - div, sigma=vol, Fwd_Time=maturity)





            # call_price_van.append (sign_call * gear_call * c_price_van * nominal_)
            # put_price_van .append (sign_put  * gear_put  * p_price_van * nominal_)
            # fwd_price     .append(sign_call * c_price_fwd + sign_put * p_price_fwd)



            # d_call_van = (c_price_van_up - c_price_van_do) / (2*h)
            # d_put_van  = (p_price_van_up - p_price_van_do) / (2*h)
            # delta_call_van .append (sign_call * gear_call* d_call_van * nominal_)
            # delta_put_van .append (sign_put  * gear_put  *d_put_van * nominal_)

        call_price_.append(np.sum(call_price) / (gear))
        put_price_.append(np.sum(put_price) / (gear))
        strip_price_.append(np.sum(strip_price) / (gear))
        # fwd_price_.append(np.sum(fwd_price) / (len(fwd_price)))

        # call_price_van_.append(np.sum(call_price_van) / (gear_call))
        # put_price_van_.append(np.sum(put_price_van) / (gear_put))

        delta_call_.append(np.sum(delta_call) / (gear_call))
        delta_put_.append(np.sum(delta_put) / (gear_put))
        delta_strip_.append(np.sum(delta_strip) / (gear))

        # delta_call_van_.append(np.sum(delta_call_van) / (gear_call))
        # delta_put_van_.append(np.sum(delta_put_van) / (gear_put))

        call_price     = []
        put_price      = []
        # call_price_van = []
        # put_price_van  = []
        strip_price    = []

        delta_call   = []
        delta_put    = []
        delta_strip  = []

        # delta_call_van   = []
        # delta_put_van    = []

        # fwd_price   = []

# --- Interactive chart with tabs ---
with container_2d:
    # st.markdown("### Interactive Pricing Panels")

    # --- Tabs for Sensitivity and Margin ---
    tab_sensitivity, tab_margin = st.tabs(["Sensitivity", "Margin"])

    # --- Sensitivity / MtM Tab ---
    with tab_sensitivity:
        # Toggle to show individual legs
        show_legs = st.toggle(
            "Show Long / Short Legs", 
            value=False, 
            key="show_legs_toggle", 
            help="Toggle to display individual legs alongside the combined MtM profile."
        )

        fig = go.Figure()

        # Combined MtM
        fig.add_trace(go.Scatter(
            x=spots,
            y=strip_price_,
            mode='lines',
            name=f"{type_} MtM",
            line=dict(color='cyan', width=3),
            hovertemplate="<b>MtM:</b> %{y:,.2f}<extra></extra>"
        ))

        # Individual legs
        if show_legs:
            fig.add_trace(go.Scatter(
                x=spots,
                y=call_price_,
                mode='lines',
                name="Long Leg (Call)" if type_ == "Accumulator" else "Short Leg (Call)",
                line=dict(color='lime', width=2, dash='dot'),
                hovertemplate="<b>Call Leg:</b> %{y:,.2f}<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=spots,
                y=put_price_,
                mode='lines',
                name="Short Leg (Put)" if type_ == "Accumulator" else "Long Leg (Put)",
                line=dict(color='magenta', width=2, dash='dot'),
                hovertemplate="<b>Put Leg:</b> %{y:,.2f}<extra></extra>"
            ))

        # Reference lines
        fig.add_vline(x=strike, line=dict(color="white", dash="dash"),
                      annotation_text="Strike", annotation_position="top")
        fig.add_vline(x=barrier, line=dict(color="orange", dash="dash"),
                      annotation_text="Barrier", annotation_position="top")

        position_mtm_legend = dict(orientation="v", yanchor="bottom", y=-0.00, xanchor="right", x=1) if type_ == "Accumulator" else dict(orientation="v", yanchor="bottom", y=-0.0, xanchor="left", x=0)


        fig.update_layout(
            title=f"{type_} MtM Profile",
            xaxis_title="Spot",
            yaxis_title="MtM",
            template="plotly_dark",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=50, b=0),
            legend= position_mtm_legend
        )

        st.plotly_chart(fig, use_container_width=True)

    # --- Margin Tab ---
    with tab_margin:
        # shock = st.number_input(
        #     "Margin Shock", 0.01, 0.1, 0.2,
        #     help="Specify shock level for worst MtM drop calculation"
        # )

        if periods == 1 and periods_guaranteed == 1:
            strike_notional = np.where(spots_ <= barrier, strike * nominal , 0) if type_ == 'Accumulator' else np.where(spots_ <= barrier, 0, strike * nominal )
            spot_notional = np.where(spots_ <= barrier, spots_ * nominal , 0) if type_ == 'Accumulator' else np.where(spots_ <= barrier, 0, spots_ * nominal )
        else:
            strike_notional = np.where(spots_ <= barrier, strike * nominal * gear, 0) if type_ == 'Accumulator' else np.where(spots_ <= barrier, 0, strike * nominal * gear)
            spot_notional = np.where(spots_ <= barrier, spots_ * nominal * gear, 0) if type_ == 'Accumulator' else np.where(spots_ <= barrier, 0, spots_ * nominal * gear)

        kod_mtm = dict(zip(spots, strip_price_))
        kod_delta = dict(zip(spots, delta_strip_))

        true_delta = {}
        proxy_delta = {}

        for s in spots_:
            try:
                coeff = kod_delta[s]

                tangent = kod_mtm[s] + coeff * (spots - s)
                tangent = dict(zip(spots, tangent))
                tangent_plus = tangent[s * (1 + shock)]
                tangent_minus = tangent[s * (1 - shock)]

                true_delta[s] = -1 * min(kod_mtm[s * (1 + shock)] - kod_mtm[s],
                                        kod_mtm[s * (1 - shock)] - kod_mtm[s])
                proxy_delta[s] = -1 * min(tangent_plus - tangent[s],
                                        tangent_minus - tangent[s])

            except Exception as e:
                print(e)

        x_vals = np.array(list(true_delta.keys()))
        y_true = np.array(list(true_delta.values()))
        y_proxy = np.array(list(proxy_delta.values()))

        # Compute % diff wrt Actual
        pct_diff_proxy = np.round(100 * (y_proxy - y_true) / y_true, 2)
        pct_diff_strike = np.round(100 * ((strike_notional * shock) - y_true) / y_true, 2)
        pct_diff_spot = np.round(100 * ((spot_notional * shock) - y_true) / y_true, 2)

        fig = go.Figure()

        # Actual
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_true,
            mode='lines',
            name='Actual',
            line=dict(color='cyan', width=2),
            hovertemplate="<b>Actual:</b> %{y:,.0f}<extra></extra>"
        ))

        # Δ Proxy
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_proxy,
            mode='lines',
            name='Δ Proxy',
            line=dict(color='orange', width=2),
            customdata=np.stack([pct_diff_proxy], axis=-1),
            hovertemplate="<b>Δ Proxy:</b> %{y:,.0f}<br>% From Actual: %{customdata[0]:+.2f}%<extra></extra>"
        ))

        # Strike Notional
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=strike_notional * shock,
            mode='lines',
            name='Strike Notional',
            line=dict(color='magenta', width=1, dash='dot'),
            customdata=np.stack([pct_diff_strike], axis=-1),
            hovertemplate="<b>Strike Notional:</b> %{y:,.0f}<br>% From Actual: %{customdata[0]:+.2f}%<extra></extra>"
        ))

        # Spot Notional
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=spot_notional * shock,
            mode='lines',
            name='Spot Notional',
            line=dict(color='lime', width=1, dash='dot'),
            customdata=np.stack([pct_diff_spot], axis=-1),
            hovertemplate="<b>Spot Notional:</b> %{y:,.0f}<br>% From Actual: %{customdata[0]:+.2f}%<extra></extra>"
        ))

        # Reference lines
        fig.add_vline(x=strike, line=dict(color="white", dash="dash"),
                    annotation_text="Strike", annotation_position="top")
        fig.add_vline(x=barrier, line=dict(color="red", dash="dash"),
                    annotation_text="Barrier", annotation_position="top")

        legend_position = dict(orientation="v", yanchor="top", y=1.02, xanchor="right", x=1) if type_ == "Accumulator" else dict(orientation="v", yanchor="top", y=1.02, xanchor="left", x=0)

        fig.update_layout(
            title=f"Margin [{shock*100:.1f}% Risk Factor]",
            xaxis_title="Spot",
            yaxis_title="Worst MtM Drop",
            template="plotly_dark",
            hovermode="x unified",
            hoverlabel=dict(bgcolor="black", font_size=12, font_color="white"),
            margin=dict(l=0, r=0, t=50, b=0),
            legend=legend_position
        )

        st.plotly_chart(fig, use_container_width=True)


    st.markdown('</div>', unsafe_allow_html=True)








