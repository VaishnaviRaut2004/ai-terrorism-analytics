# step8_ai_dashboard_v2.py
# AI-Powered Terrorism Analytics — PIN Auth + Full Dashboard (Stable)
# Author: Redesigned for Vaishnavi Raut
# Run: streamlit run step8_ai_dashboard_v2.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import pycountry

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="AI Terrorism Analytics — Premium", layout="wide", page_icon="🌍")

# -------------------------
# PIN AUTH (Option 2) - Simple, stable, no external auth libs
# -------------------------
# NOTE: For production, store PINs securely (not in plain text).
USERS = {
    "vaishnavi": {"name": "Vaishnavi Raut", "pin": "1981"},
    "demo": {"name": "Demo User", "pin": "0000"}
}

# Initialize session state keys safely
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False
if "username" not in st.session_state:
    st.session_state.username = None
if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0
if "login_message" not in st.session_state:
    st.session_state.login_message = ""

def login_widget():
    """Render login fields and handle login without experimental_rerun."""
    st.markdown(
        """
        <div style="display:flex; gap:12px; align-items:center;">
          <h2 style="margin:0; color:#E6B85A;">🔒 Login</h2>
          <div style="color:#9aa6ad; margin-left:6px; font-size:13px;">(Use your username and 4-digit PIN)</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    col_user, col_pin, col_btn = st.columns([2,1,1])
    user_input = st.text_input("Username", key="login_user_input", placeholder="vaishnavi")
    pin_input = st.text_input("PIN", key="login_pin_input", type="password", placeholder="1981")

    if st.button("Login", key="login_button"):
        st.session_state.login_attempts += 1
        # check credentials
        if user_input in USERS and pin_input == USERS[user_input]["pin"]:
            st.session_state.auth_ok = True
            st.session_state.username = user_input
            st.session_state.login_message = f"Welcome, {USERS[user_input]['name']}!"
            st.success(st.session_state.login_message)
            # do NOT call experimental_rerun; simply proceed — dashboard will render because auth_ok True
        else:
            st.session_state.login_message = "Invalid username or PIN."
            st.error(st.session_state.login_message)
            if st.session_state.login_attempts >= 5:
                st.warning("Too many failed attempts. Please restart the app or try later.")
    st.markdown("---")

def logout():
    st.session_state.auth_ok = False
    st.session_state.username = None
    st.session_state.login_message = ""
    # do not call rerun; page will now show login on next interaction

# -------------------------
# If not logged in: show login panel and stop
# -------------------------
if not st.session_state.auth_ok:
    st.markdown(
        """
        <style>
        .login-box { max-width:820px; margin:30px auto; padding:18px; border-radius:12px;
                    background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.02));
                    border:1px solid rgba(255,255,255,0.03); box-shadow: 0 10px 30px rgba(0,0,0,0.6);}
        .login-note { color:#c0c9d6; margin-top:6px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='login-box'>", unsafe_allow_html=True)
    st.markdown("<h1 style='color:#E6B85A; margin:0;'>🌍 AI-Powered Terrorism Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p class='login-note'>Premium dashboard — enter credentials to continue.</p>", unsafe_allow_html=True)
    login_widget()
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# -------------------------
# AUTHENTICATED: show dashboard
# -------------------------
# Header / topbar (CSS)
st.markdown(
    """
    <style>
    :root{
      --bg1: #05060a;
      --bg2: #0b1420;
      --glass: rgba(255,255,255,0.03);
      --gold: #E6B85A;
      --muted: #9aa6ad;
    }
    .stApp { background: linear-gradient(180deg,var(--bg1), var(--bg2)); color: #e9f4ff; font-family: Inter, sans-serif; }
    .topbar { padding:12px 18px; margin-bottom:16px; border-radius:12px; background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); display:flex; justify-content:space-between; align-items:center; box-shadow: 0 6px 30px rgba(0,0,0,0.6);}
    .chip { background: rgba(255,255,255,0.02); padding:6px 10px; border-radius:999px; color:var(--muted); font-size:12px; border:1px solid rgba(255,255,255,0.02); }
    .card { background: var(--glass); border-radius:12px; padding:12px; margin-bottom:12px; border:1px solid rgba(255,255,255,0.03); box-shadow: 0 8px 30px rgba(0,0,0,0.5);}
    .metric { padding:10px; border-radius:8px; }
    .small { color:var(--muted); font-size:12px; }
    footer {visibility:hidden;}
    #MainMenu {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"""
<div class="topbar">
  <div>
    <h2 style="margin:0; color:#E6B85A;">🌍 AI-Powered Terrorism Analytics — Premium</h2>
    <div style="color:#9aa6ad; font-size:13px;">Interactive dashboard • {USERS.get(st.session_state.username, {}).get('name','User')}</div>
  </div>
  <div style="display:flex; gap:10px; align-items:center;">
    <div class="chip">Data: gti_cleaned.csv</div>
    <div class="chip">Theme: Black + Gold</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Logout button (unique key)
if st.button("Logout", key="logout_button_v2"):
    logout()

# -------------------------
# LOAD DATA
# -------------------------
DATA_FILE = "gti_cleaned.csv"
if not Path(DATA_FILE).exists():
    st.error("❌ Dataset not found. Place 'gti_cleaned.csv' in the same folder as this app.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path)
    return df

data = load_data(DATA_FILE)

# Column normalization and safety
if "Country" not in data.columns and "country" in data.columns:
    data.rename(columns={"country": "Country"}, inplace=True)

def safe_iso3_from_name(name):
    if pd.isna(name):
        return None
    s = str(name).strip()
    if len(s) == 3 and s.isalpha():
        return s.upper()
    try:
        c = pycountry.countries.lookup(s)
        return getattr(c, "alpha_3", None)
    except Exception:
        return None

if "iso3c" not in data.columns:
    if "Country" in data.columns:
        data["iso3c"] = data["Country"].apply(safe_iso3_from_name)
    else:
        data["iso3c"] = None
else:
    data["iso3c"] = data["iso3c"].astype(str).str.strip().replace({"nan": None, "None": None})
    missing_mask = data["iso3c"].isna() | (data["iso3c"] == "")
    if "Country" in data.columns and missing_mask.any():
        data.loc[missing_mask, "iso3c"] = data.loc[missing_mask, "Country"].apply(safe_iso3_from_name)

# numeric conversions & fills
for c in ["Incidents", "Fatalities", "Injuries", "Hostages", "Score", "Rank", "Year"]:
    if c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")
for c in ["Incidents", "Fatalities", "Injuries", "Hostages", "Score", "Rank"]:
    if c in data.columns:
        data[c] = data[c].fillna(0)

# -------------------------
# SIDEBAR CONTROLS (unique keys)
# -------------------------
with st.sidebar:
    st.header("Filters & Settings")
    years = []
    if "Year" in data.columns:
        years = sorted(data["Year"].dropna().astype(int).unique().tolist())
    selected_year = st.selectbox("Year", years, index=len(years)-1 if years else 0, key="sidebar_year_select_v2")
    country_list = ["All"]
    if "Country" in data.columns:
        country_list += sorted(data["Country"].dropna().unique().tolist())
    selected_country = st.selectbox("Country", country_list, index=0, key="sidebar_country_select_v2")
    st.markdown("---")
    st.subheader("Display Options")
    show_trend = st.checkbox("Show multi-year trend", value=True, key="sidebar_trend_check_v2")
    show_3d = st.checkbox("Enable 3D scatter (may be slower)", value=False, key="sidebar_3d_check_v2")
    st.markdown("---")
    st.subheader("Model")
    retrain = st.checkbox("Retrain full model (slow)", value=False, key="sidebar_retrain_check_v2")
    st.markdown("<div class='small'>Tip: retrain only if data/parameters changed.</div>", unsafe_allow_html=True)

# apply filters safely
df = data.copy()
if "Year" in df.columns and selected_year is not None:
    df = df[df["Year"] == int(selected_year)]
if selected_country != "All" and "Country" in df.columns:
    df = df[df["Country"] == selected_country]

# -------------------------
# TOP METRICS
# -------------------------
col1, col2, col3, col4 = st.columns([2,1,1,1])
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    try:
        main_country = data.groupby("Country")["Incidents"].sum().idxmax() if "Country" in data.columns else "N/A"
    except Exception:
        main_country = "N/A"
    st.markdown(f"<div style='font-size:16px; font-weight:700; color:#E6B85A'>{main_country}</div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Most affected country (all years)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    try:
        deadliest_year = int(data.groupby("Year")["Fatalities"].sum().idxmax()) if "Year" in data.columns else 0
    except Exception:
        deadliest_year = 0
    st.markdown(f"<div style='font-size:16px; font-weight:600'>{deadliest_year}</div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Deadliest year</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    avg_score = data["Score"].mean() if "Score" in data.columns else 0
    st.markdown(f"<div style='font-size:16px; font-weight:600'>{avg_score:.2f}</div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Average Score</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    country_count = data["Country"].nunique() if "Country" in data.columns else 0
    st.markdown(f"<div style='font-size:16px; font-weight:600'>{country_count}</div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Countries analyzed</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# -------------------------
# MAIN LAYOUT: map, trends, side charts
# -------------------------
left, right = st.columns((2,1))
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(f"🗺️ Global Terrorism Score — {selected_year}")
    if {"Country", "iso3c"}.issubset(df.columns):
        map_df = df.dropna(subset=["iso3c"]).groupby(["Country","iso3c"], as_index=False).agg({"Score":"mean","Fatalities":"sum","Incidents":"sum"})
    else:
        map_df = pd.DataFrame()
    if map_df.empty:
        st.info("No map data for selected filters.")
    else:
        fig_map = px.choropleth(
            map_df,
            locations="iso3c",
            color="Score",
            hover_name="Country",
            hover_data={"Fatalities":True, "Incidents":True, "iso3c":False},
            color_continuous_scale=px.colors.sequential.Plasma,
            projection="natural earth"
        )
        fig_map.update_layout(margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_map, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if show_trend:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📈 Multi-Year Trend")
        if "Year" in data.columns:
            trend_df = data.groupby("Year").agg({"Incidents":"sum","Fatalities":"sum","Score":"mean"}).reset_index()
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=trend_df["Year"], y=trend_df["Incidents"], name="Incidents", mode="lines+markers", line=dict(width=2, color="#E6B85A")))
            fig_trend.add_trace(go.Scatter(x=trend_df["Year"], y=trend_df["Fatalities"], name="Fatalities", mode="lines+markers", line=dict(width=2, color="#8c92ac")))
            fig_trend.update_layout(height=360, margin=dict(t=10,b=0,l=0,r=0), legend=dict(orientation="h"), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Year column not available.")
        st.markdown("</div>", unsafe_allow_html=True)

    if show_3d:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🌐 3D Scatter — Sampled")
        if len(data) == 0:
            st.info("No data for 3D plot.")
        else:
            sample_n = min(1200, len(data))
            sample = data.sample(sample_n, random_state=42)
            if {"Incidents","Fatalities","Score"}.issubset(sample.columns):
                fig3d = px.scatter_3d(sample, x="Incidents", y="Fatalities", z="Score", color="Country" if "Country" in sample.columns else None, size="Injuries" if "Injuries" in sample.columns else None, opacity=0.8)
                fig3d.update_layout(height=600, margin=dict(t=10,b=0,l=0,r=0), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig3d, use_container_width=True)
            else:
                st.info("Missing columns for 3D scatter.")
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏆 Top 10 Countries by Fatalities")
    if "Country" in df.columns and "Fatalities" in df.columns:
        top10 = df.groupby("Country")["Fatalities"].sum().nlargest(10).reset_index()
    else:
        top10 = pd.DataFrame()
    if top10.empty:
        st.info("No data for selected filters.")
    else:
        bar_fig = px.bar(top10.sort_values("Fatalities"), x="Fatalities", y="Country", orientation="h", color="Fatalities", color_continuous_scale=px.colors.sequential.Reds)
        bar_fig.update_layout(height=360, margin=dict(t=0,b=0,l=0,r=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(bar_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔬 Correlation Heatmap")
    numeric_cols = [c for c in ["Incidents","Fatalities","Injuries","Hostages","Score","Rank"] if c in data.columns]
    if len(numeric_cols) >= 2:
        corr = data[numeric_cols].corr()
        corr_fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale=px.colors.diverging.RdBu)
        corr_fig.update_layout(height=320, margin=dict(t=0,b=0,l=0,r=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(corr_fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns to compute correlation.")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# PREDICTION PANEL (unique widget keys)
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🤖 Quick Score Estimator")

colp1, colp2 = st.columns([1,1])
with colp1:
    i = st.number_input("Incidents", min_value=0, max_value=100000, value=200, key="pred_incidents_v2")
    f = st.number_input("Fatalities", min_value=0, max_value=100000, value=50, key="pred_fatalities_v2")
    inj = st.number_input("Injuries", min_value=0, max_value=100000, value=100, key="pred_injuries_v2")
with colp2:
    h = st.number_input("Hostages", min_value=0, max_value=10000, value=5, key="pred_hostages_v2")
    r = st.slider("Rank", 1, 300, 50, key="pred_rank_v2")

feature_cols = ["Incidents","Fatalities","Injuries","Hostages","Rank"]
X_all = data.reindex(columns=feature_cols).fillna(0)
y_all = data["Score"].fillna(0) if "Score" in data.columns else pd.Series(np.zeros(len(X_all)))

model = RandomForestRegressor(n_estimators=150, random_state=42)
model_trained = False
if len(X_all) == 0 or len(y_all) == 0:
    st.warning("Not enough data to train the prediction model.")
else:
    try:
        if retrain:
            with st.spinner("Training model on full dataset..."):
                model.fit(X_all, y_all)
                model_trained = True
                st.success("Model trained on full dataset.")
        else:
            sample_n = min(2500, len(X_all))
            with st.spinner("Training small model for quick predictions..."):
                Xs = X_all.sample(sample_n, random_state=42)
                ys = y_all.loc[Xs.index]
                model.fit(Xs, ys)
                model_trained = True
    except Exception as e:
        st.error(f"Model training failed: {e}")

if st.button("Predict Score", key="predict_button_v2"):
    if not model_trained:
        st.error("Model not trained. Toggle retrain or try again.")
    else:
        pred = model.predict([[i, f, inj, h, r]])[0]
        st.markdown(f"<div style='font-size:20px; color:#E6B85A; font-weight:600'>🌋 Predicted Score: {pred:.2f}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# AUTO INSIGHTS
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧠 Auto Insights")
ia, ib, ic = st.columns(3)
with ia:
    top_country_sel = df.groupby("Country")["Incidents"].sum().idxmax() if (not df.empty and "Country" in df.columns) else "N/A"
    st.markdown(f"**Top (selected year)**\n\n{top_country_sel}")
with ib:
    spike_year = int(data.groupby("Year")["Incidents"].sum().idxmax()) if "Year" in data.columns else "N/A"
    st.markdown(f"**Year with spike (incidents)**\n\n{spike_year}")
with ic:
    avg_fatal = data["Fatalities"].mean() if "Fatalities" in data.columns else 0
    st.markdown(f"**Avg Fatalities (all years)**\n\n{avg_fatal:.1f}")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# FOOTER
# -------------------------
st.markdown("<div style='text-align:center; color:#c0c9d6; padding:10px; font-size:13px;'>✨ © 2025 Vaishnavi Raut — AI-Powered Terrorism Analytics</div>", unsafe_allow_html=True)
