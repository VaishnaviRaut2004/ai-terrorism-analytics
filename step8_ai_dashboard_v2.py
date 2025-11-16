# step8_ai_dashboard_v2.py
# Sunset-inspired UI (no theme name shown) — Role-based dashboard
# Admin (edit) / Viewer (interactive read-only)
# Run: streamlit run step8_ai_dashboard_v2.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import pycountry
import time

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Terrorism Analytics Dashboard", layout="wide", page_icon="📊")

# -------------------------
# Credentials (PIN-based)
# -------------------------
USERS = {
    "vaishnavi": {"name": "Vaishnavi Raut", "pin": "1981", "role": "admin"},
    "viewer": {"name": "Viewer", "pin": "0000", "role": "viewer"},
}

# -------------------------
# Session init
# -------------------------
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False
if "username" not in st.session_state:
    st.session_state.username = None
if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0

# -------------------------
# Login widget (gradient background)
# -------------------------
def login_widget():
    st.markdown(
        """
        <div style="display:flex; gap:12px; align-items:center;">
          <h2 style="margin:0; color:#FFF7F2;">🔐 Sign in</h2>
          <div style="color:#FFD9E6; margin-left:6px; font-size:13px;">(username & PIN)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([2, 1])
    with col1:
        username_input = st.text_input("Username", key="login_user", placeholder="vaishnavi")
    with col2:
        pin_input = st.text_input("PIN", key="login_pin", type="password", placeholder="1981")

    if st.button("Login", key="login_btn"):
        st.session_state.login_attempts += 1
        if username_input in USERS and pin_input == USERS[username_input]["pin"]:
            st.session_state.auth_ok = True
            st.session_state.username = username_input
            st.success(f"Welcome, {USERS[username_input]['name']} ({USERS[username_input]['role'].upper()})")
            time.sleep(0.4)
        else:
            st.error("Invalid username or PIN.")
            if st.session_state.login_attempts >= 5:
                st.warning("Too many failed attempts. Refresh to try again.")

def logout():
    st.session_state.auth_ok = False
    st.session_state.username = None

# Show login if not authenticated (with gradient)
if not st.session_state.auth_ok:
    st.markdown(
        """
        <style>
        .login-outer {
            min-height:480px;
            display:flex;
            align-items:center;
            justify-content:center;
            padding:40px 16px;
            background: radial-gradient(circle at 10% 20%, rgba(255,111,60,0.12), transparent 8%),
                        linear-gradient(135deg, #2A0A3A 0%, #8A3FFC 35%, #FF5C9E 65%, #FF6F3C 100%);
        }
        .login-card {
            width:920px;
            border-radius:14px;
            padding:28px;
            backdrop-filter: blur(6px);
            background: rgba(255,247,242,0.04);
            border: 1px solid rgba(255,255,255,0.04);
            box-shadow: 0 12px 40px rgba(10,10,20,0.6);
        }
        .login-title { font-size:32px; color: #FFF7F2; margin:0; font-weight:300; }
        .login-sub { color:#FFD9E6; margin-top:6px; font-size:14px; }
        </style>
        <div class="login-outer">
          <div class="login-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <div>
                <h1 class="login-title">Terrorism Analytics Dashboard</h1>
                <div class="login-sub">Secure access — sign in to continue</div>
              </div>
            </div>
            <hr style="border:0.5px solid rgba(255,255,255,0.03); margin:14px 0;">
        """,
        unsafe_allow_html=True,
    )
    login_widget()
    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()

# -------------------------
# Post-login role
# -------------------------
current_user = st.session_state.username
role = USERS.get(current_user, {}).get("role", "viewer")

# -------------------------
# Global CSS (large, light fonts and aesthetic)
# -------------------------
st.markdown(
    """
    <style>
    :root{
      --bg1: #2A0A3A;
      --bg2: #8A3FFC;
      --accentA: #FF6F3C;
      --accentB: #FF5C9E;
      --muted: #FFD9E6;
    }
    .stApp { background: linear-gradient(180deg, #120215, #1c0530); color: #FFF7F2; font-family: Inter, ui-sans-serif, system-ui; }
    .topbar {
      padding:14px 20px; margin-bottom:14px; border-radius:10px;
      background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      border:1px solid rgba(255,255,255,0.02);
    }
    h1.main-title { font-size:32px; font-weight:300; margin:0; color:#FFF7F2; }
    .muted { color: #FFD9E6; font-size:14px; }
    .card { background: rgba(255,247,242,0.03); padding:14px; border-radius:12px; margin-bottom:12px; border:1px solid rgba(255,255,255,0.02); }
    .metric-title { font-size:18px; font-weight:600; margin:0; color:#FFF7F2; }
    .metric-sub { font-size:13px; color:#FFD9E6; }
    .small { font-size:13px; color:#FFD9E6; }
    .btn { background: linear-gradient(90deg, var(--accentA), var(--accentB)); color:#210215; padding:8px 12px; border-radius:10px; font-weight:600; border:none; }
    footer { color:#FFD9E6; padding:10px; text-align:center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Topbar (generic title, no theme name)
# -------------------------
st.markdown(
    f"""
    <div class="topbar">
      <div style="display:flex; flex-direction:column;">
        <h1 class="main-title">Terrorism Analytics Dashboard</h1>
        <div class="muted">User: <b>{USERS[current_user]['name']}</b> • Role: <b>{role.upper()}</b></div>
      </div>
      <div style="display:flex; gap:12px; align-items:center;">
        <div class="small">Data file: gti_cleaned.csv</div>
        <div class="small">Prepared for reporting & analysis</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Logout
if st.button("Logout", key="logout_btn"):
    logout()
    st.experimental_rerun()

# -------------------------
# Load data (safe)
# -------------------------
DATA_FILE = "gti_cleaned.csv"
if not Path(DATA_FILE).exists():
    st.error("❌ Dataset 'gti_cleaned.csv' not found. Upload it to the app folder.")
    st.stop()

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

data = load_data(DATA_FILE)

# -------------------------
# Ensure iso3c and numeric columns are safe
# -------------------------
def safe_iso3(name):
    try:
        if pd.isna(name):
            return None
        s = str(name).strip()
        if len(s) == 3 and s.isalpha():
            return s.upper()
        return pycountry.countries.lookup(s).alpha_3
    except Exception:
        return None

if "iso3c" not in data.columns or data["iso3c"].isnull().any():
    if "Country" in data.columns:
        data["iso3c"] = data["Country"].apply(safe_iso3)
    else:
        data["iso3c"] = None

for c in ["Incidents", "Fatalities", "Injuries", "Hostages", "Score", "Rank", "Year"]:
    if c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")
for c in ["Incidents", "Fatalities", "Injuries", "Hostages", "Score", "Rank"]:
    if c in data.columns:
        data[c] = data[c].fillna(0)

# -------------------------
# Sidebar (fixed dropdown lists for Year & Country) - viewers can select
# -------------------------
with st.sidebar:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Filters & Settings")
    years = sorted(list(map(int, data["Year"].dropna().unique()))) if "Year" in data.columns else []
    if years:
        selected_year = st.selectbox("Year", years, index=len(years) - 1, key="sel_year", disabled=False)
    else:
        selected_year = None

    countries = sorted(data["Country"].dropna().unique().tolist()) if "Country" in data.columns else []
    country_options = ["All"] + countries
    selected_country = st.selectbox("Country", country_options, index=0, key="sel_country", disabled=False)

    st.markdown("---")
    st.subheader("Visual Options")
    if role == "admin":
        top_n = st.slider("Top N countries", 5, 30, 10, key="top_n")
        show_anim = st.checkbox("Animated map (year)", value=False, key="show_anim")
        show_bubble = st.checkbox("Bubble chart", value=True, key="show_bubble")
        show_heat = st.checkbox("Heatmap", value=True, key="show_heat")
        show_pie = st.checkbox("Show pie (attack types)", value=False, key="show_pie")
    else:
        # viewers can view but not change visualization defaults (top_n fixed read-only visually)
        top_n = st.slider("Top N countries", 5, 30, 10, key="top_n_view", disabled=True)
        show_anim = st.checkbox("Animated map (year)", value=False, key="show_anim_view", disabled=True)
        show_bubble = st.checkbox("Bubble chart", value=True, key="show_bubble_view", disabled=True)
        show_heat = st.checkbox("Heatmap", value=True, key="show_heat_view", disabled=True)
        show_pie = st.checkbox("Show pie (attack types)", value=False, key="show_pie_view", disabled=True)

    st.markdown("---")
    st.subheader("Model Controls")
    if role == "admin":
        retrain = st.checkbox("Retrain model (slow)", value=False, key="retrain")
        n_estim = st.number_input("RF n_estimators", min_value=10, max_value=500, value=150, key="n_estim")
    else:
        retrain = False
        n_estim = 150
        st.markdown("<div class='small'>Viewer: model controls are disabled.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Apply filters to df
# -------------------------
df = data.copy()
if selected_year is not None and "Year" in df.columns:
    try:
        df = df[df["Year"] == int(selected_year)]
    except Exception:
        pass
if selected_country and selected_country != "All":
    df = df[df["Country"] == selected_country]

# -------------------------
# Top metrics (large, light font)
# -------------------------
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    try:
        top_country_all = data.groupby("Country")["Incidents"].sum().idxmax()
    except Exception:
        top_country_all = "N/A"
    st.markdown(f"<div class='metric-title'>{top_country_all}</div><div class='metric-sub'>Most affected country (all years)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    try:
        deadliest_year = int(data.groupby("Year")["Fatalities"].sum().idxmax())
    except Exception:
        deadliest_year = "N/A"
    st.markdown(f"<div class='metric-title'>{deadliest_year}</div><div class='metric-sub'>Deadliest year</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    avg_score = data["Score"].mean() if "Score" in data.columns else 0
    st.markdown(f"<div class='metric-title'>{avg_score:.2f}</div><div class='metric-sub'>Average Score</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with col4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    country_count = data["Country"].nunique() if "Country" in data.columns else 0
    st.markdown(f"<div class='metric-title'>{country_count}</div><div class='metric-sub'>Countries analyzed</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# -------------------------
# Main layout: map and visualizations
# -------------------------
left, right = st.columns((2, 1))
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(f"🗺️ Global Terrorism — {selected_year if selected_year else 'All Years'}")

    try:
        if show_anim and {"Year", "iso3c", "Score"}.issubset(data.columns):
            anim_df = data.groupby(["Year", "Country", "iso3c"], as_index=False).agg({"Score": "mean", "Incidents": "sum", "Fatalities": "sum"})
            anim_df = anim_df.sort_values("Year")
            fig_map = px.choropleth(anim_df, locations="iso3c", color="Score", hover_name="Country",
                                    animation_frame="Year", color_continuous_scale=px.colors.sequential.Plasma, projection="natural earth")
            fig_map.update_layout(margin=dict(t=10, b=0, l=0, r=0))
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            map_df = df.groupby(["Country", "iso3c"], as_index=False).agg({"Score": "mean", "Incidents": "sum", "Fatalities": "sum"}) if {"Country", "iso3c"}.issubset(df.columns) else pd.DataFrame()
            if map_df.empty:
                st.info("No map data for selected filters.")
            else:
                fig = px.choropleth(map_df, locations="iso3c", color="Score", hover_name="Country", color_continuous_scale=px.colors.sequential.Tealgrn, projection="natural earth")
                fig.update_layout(margin=dict(t=10, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Map failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Multi-year trend (dual axis)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Incidents vs Fatalities (Multi-year)")
    if "Year" in data.columns:
        trend = data.groupby("Year").agg({"Incidents": "sum", "Fatalities": "sum", "Score": "mean"}).reset_index()
        fig_tr = go.Figure()
        fig_tr.add_trace(go.Bar(x=trend["Year"], y=trend["Incidents"], name="Incidents", marker_color="#FF9A6A"))
        fig_tr.add_trace(go.Scatter(x=trend["Year"], y=trend["Fatalities"], name="Fatalities", yaxis="y2", marker_color="#8A3FFC"))
        fig_tr.update_layout(
            yaxis=dict(title="Incidents"),
            yaxis2=dict(title="Fatalities", overlaying="y", side="right"),
            legend=dict(orientation="h"),
            margin=dict(t=10, b=0),
        )
        st.plotly_chart(fig_tr, use_container_width=True)
    else:
        st.info("Year column not available.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Bubble chart
    if show_bubble:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔵 Fatalities vs Injuries (bubble size = Incidents)")
        if {"Fatalities", "Injuries", "Incidents", "Country"}.issubset(data.columns):
            bubble_df = df.groupby("Country", as_index=False).agg({"Fatalities": "sum", "Injuries": "sum", "Incidents": "sum"})
            bubble_df = bubble_df[bubble_df["Fatalities"] + bubble_df["Injuries"] > 0]
            if bubble_df.empty:
                st.info("No bubble data for selected filters.")
            else:
                fig_b = px.scatter(bubble_df, x="Injuries", y="Fatalities", size="Incidents", hover_name="Country", size_max=40, color="Incidents", color_continuous_scale=px.colors.sequential.OrRd)
                st.plotly_chart(fig_b, use_container_width=True)
        else:
            st.info("Required columns missing for bubble chart.")
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏆 Top Countries by Fatalities")
    if "Country" in df.columns and "Fatalities" in df.columns:
        top10 = df.groupby("Country")["Fatalities"].sum().nlargest(top_n).reset_index()
        if top10.empty:
            st.info("No data for selected filters.")
        else:
            fig_bar = px.bar(top10.sort_values("Fatalities"), x="Fatalities", y="Country", orientation="h", color="Fatalities", color_continuous_scale=px.colors.sequential.OrRd)
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No data for this chart.")
    st.markdown("</div>", unsafe_allow_html=True)

    if show_heat:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔥 Heatmap — Year vs Top Countries (Incidents)")
        if "Year" in data.columns and "Country" in data.columns:
            top_countries = data.groupby("Country")["Incidents"].sum().nlargest(top_n).index.tolist()
            heat_df = data[data["Country"].isin(top_countries)].pivot_table(values="Incidents", index="Country", columns="Year", aggfunc="sum").fillna(0)
            if heat_df.empty:
                st.info("No data for heatmap.")
            else:
                fig_h = px.imshow(heat_df, text_auto=True, aspect="auto", color_continuous_scale=px.colors.sequential.Magma)
                st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("Year/Country missing.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Correlation
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔬 Correlation (Numeric Features)")
    numeric_cols = [c for c in ["Incidents", "Fatalities", "Injuries", "Hostages", "Score", "Rank"] if c in data.columns]
    if len(numeric_cols) >= 2:
        corr = data[numeric_cols].corr()
        corr_fig = px.imshow(corr, text_auto=True, color_continuous_scale=px.colors.diverging.RdBu)
        st.plotly_chart(corr_fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation.")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Prediction panel (role-aware)
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🤖 Score Predictor")

for c in ["Incidents", "Fatalities", "Injuries", "Hostages", "Rank"]:
    if c not in data.columns:
        data[c] = 0

X = data[["Incidents", "Fatalities", "Injuries", "Hostages", "Rank"]].fillna(0)
y = data["Score"].fillna(0) if "Score" in data.columns else pd.Series(np.zeros(len(X)), index=X.index)

if role == "admin":
    ca, cb = st.columns(2)
    with ca:
        inc = st.number_input("Incidents", min_value=0, max_value=100000, value=200, key="inc_admin")
        fat = st.number_input("Fatalities", min_value=0, max_value=100000, value=50, key="fat_admin")
        inj = st.number_input("Injuries", min_value=0, max_value=100000, value=100, key="inj_admin")
    with cb:
        host = st.number_input("Hostages", min_value=0, max_value=10000, value=5, key="host_admin")
        rank = st.slider("Rank", 1, 300, 50, key="rank_admin")

    if st.button("Predict (Admin)"):
        model = RandomForestRegressor(n_estimators=(n_estim if "n_estim" in locals() else 150), random_state=42)
        sample_n = min(2000, len(X))
        if sample_n > 0:
            Xs = X.sample(sample_n, random_state=42)
            ys = y.loc[Xs.index]
            model.fit(Xs, ys)
            pred = model.predict([[inc, fat, inj, host, rank]])[0]
            st.success(f"Predicted Score: {pred:.2f}")
        else:
            st.warning("Not enough data to train model.")

else:
    st.markdown("<div class='small'>Viewer mode — choose a predefined scenario to see predicted score.</div>", unsafe_allow_html=True)
    scenarios = {
        "Low (Minor incidents)": [10, 0, 1, 0, 100],
        "Medium (Localized spike)": [500, 20, 50, 2, 60],
        "High (Major spike)": [5000, 300, 800, 20, 10],
    }
    sel = st.selectbox("Scenario", list(scenarios.keys()), key="viewer_scn")
    if st.button("Show Prediction (Viewer)"):
        vals = scenarios[sel]
        model = RandomForestRegressor(n_estimators=150, random_state=42)
        sample_n = min(1500, len(X))
        if sample_n > 0:
            Xs = X.sample(sample_n, random_state=42)
            ys = y.loc[Xs.index]
            model.fit(Xs, ys)
            pred = model.predict([vals])[0]
            st.success(f"Predicted Score: {pred:.2f}")
        else:
            st.warning("Not enough data to generate prediction.")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Admin controls & download
# -------------------------
if role == "admin":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("⚙️ Admin Controls")
    csv_full = data.to_csv(index=False).encode()
    st.download_button("Download full dataset (CSV)", csv_full, file_name="gti_cleaned_full.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Filtered CSV download (all users)
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("⬇️ Export Filtered Data")
if not df.empty:
    csv_filtered = df.to_csv(index=False).encode()
    st.download_button("Download filtered CSV", csv_filtered, file_name=f"gtd_filtered_{selected_year if selected_year else 'all'}.csv", mime="text/csv")
else:
    st.info("No filtered data to download.")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Quick insights
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧠 Quick Insights")
ia, ib, ic = st.columns(3)
with ia:
    top_sel = df.groupby("Country")["Incidents"].sum().idxmax() if (not df.empty and "Country" in df.columns) else "N/A"
    st.markdown(f"**Top (selected year)**\n\n{top_sel}")
with ib:
    spike = int(data.groupby("Year")["Incidents"].sum().idxmax()) if "Year" in data.columns else "N/A"
    st.markdown(f"**Year with spike**\n\n{spike}")
with ic:
    avg_f = data["Fatalities"].mean() if "Fatalities" in data.columns else 0
    st.markdown(f"**Avg Fatalities (all years)**\n\n{avg_f:.1f}")
st.markdown("</div>", unsafe_allow_html=True)

# Footer (no theme name)
st.markdown("<div style='text-align:center; color:#FFD9E6; padding:10px;'>✨ © 2025 Vaishnavi Raut — All rights reserved</div>", unsafe_allow_html=True)
