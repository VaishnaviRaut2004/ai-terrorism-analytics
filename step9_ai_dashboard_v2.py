# step8_ai_dashboard_v2.py
# RIC Dashboard — Mixed theme (light main canvas + dark navy sidebar)
# Role-based: Admin (edit) / Viewer (read-only)
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
import io

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="RIC Terrorism Analytics",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

# -------------------------
# Simple user DB (PIN-based)
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
# Mixed theme CSS (light canvas + dark navy sidebar)
# -------------------------
MIXED_CSS = """
<style>
:root{
  --bg: #F5F7FA;         /* main background */
  --card: #FFFFFF;       /* card background */
  --text: #1F2937;       /* primary text */
  --muted: #6B7280;      /* muted text */
  --primary: #1A73E8;    /* primary blue */
  --accent: #E8F0FE;     /* accent light */
  --sidebar-bg: #0F172A; /* dark navy sidebar */
  --sidebar-ink: #E6EEF8;
  --success: #16A34A;
  --warning: #CA8A04;
  --danger: #DC2626;
}

/* Page background & fonts */
.stApp {
  background: var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto;
}

/* Topbar */
.topbar {
  padding:16px 22px;
  margin-bottom:14px;
  border-radius:10px;
  background: linear-gradient(90deg, rgba(255,255,255,0.85), rgba(255,255,255,0.95));
  border:1px solid rgba(31,41,55,0.04);
  display:flex;
  justify-content:space-between;
  align-items:center;
}

/* Titles */
.main-title { font-size:28px; font-weight:700; margin:0; color:var(--primary); }
.muted { color:var(--muted); font-size:13px; }

/* Cards */
.card {
  background: var(--card);
  padding:14px;
  border-radius:12px;
  margin-bottom:12px;
  border:1px solid rgba(31,41,55,0.06);
  box-shadow: 0 8px 24px rgba(20,30,60,0.03);
}

/* Metric styles */
.metric-title { font-size:18px; font-weight:700; margin:0; color:var(--primary); }
.metric-sub { font-size:13px; color:var(--muted); }

/* Sidebar overwrite for dark navy look */
section[data-testid="stSidebar"] > div:first-child {
  background: linear-gradient(180deg, var(--sidebar-bg), #071130);
  color: var(--sidebar-ink);
  padding: 16px 12px 18px 16px;
  border-radius: 8px;
}
section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] .stText {
  color: var(--sidebar-ink);
}

/* Sidebar headings */
section[data-testid="stSidebar"] h2 {
  color: var(--sidebar-ink);
}

/* Buttons */
.stButton>button {
  background: linear-gradient(90deg, var(--primary), #6C8CFF);
  border: none;
  color: white;
  font-weight:700;
}

/* Small helpers */
.small { font-size:13px; color:var(--muted); }

/* Login card */
.login-outer {
  min-height:420px;
  display:flex;
  align-items:center;
  justify-content:center;
  padding:30px 14px;
}
.login-card {
  width:900px;
  border-radius:12px;
  padding:28px;
  background: linear-gradient(180deg, #FFFFFF, #FBFDFF);
  border: 1px solid rgba(31,41,55,0.04);
  box-shadow: 0 18px 60px rgba(10,20,40,0.06);
}
.login-title { font-size:28px; color:var(--primary); margin:0; font-weight:800; }
.login-sub { color:var(--muted); margin-top:6px; font-size:13px; }

/* Footer */
.footer { text-align:center; color:var(--muted); padding:12px; font-size:13px; }
</style>
"""
st.markdown(MIXED_CSS, unsafe_allow_html=True)

# -------------------------
# Utility functions
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

@st.cache_data
def load_csv(path_or_buffer):
    return pd.read_csv(path_or_buffer)

def logout():
    st.session_state.auth_ok = False
    st.session_state.username = None

# -------------------------
# Login UI
# -------------------------
def login_widget():
    st.markdown(
        """
        <div style="display:flex; gap:12px; align-items:center;">
          <h2 class="login-title">🔐 Sign in</h2>
          <div class="login-sub" style="margin-left:8px;">(username & PIN)</div>
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
            time.sleep(0.35)
            st.experimental_rerun()
        else:
            st.error("Invalid username or PIN.")
            if st.session_state.login_attempts >= 5:
                st.warning("Too many failed attempts. Refresh to try again.")

# Show login card if not authenticated
if not st.session_state.auth_ok:
    st.markdown("<div class='login-outer'>", unsafe_allow_html=True)
    st.markdown("<div class='login-card'>", unsafe_allow_html=True)
    st.markdown(f"<div style='display:flex; justify-content:space-between; align-items:center;'><div><h1 class='login-title'>RIC — Terrorism Analytics</h1><div class='login-sub'>Secure access — sign in to continue</div></div></div>", unsafe_allow_html=True)
    st.markdown("<hr style='border:0.5px solid rgba(31,41,55,0.04); margin:14px 0;'>", unsafe_allow_html=True)
    login_widget()
    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()

# -------------------------
# Post-login: user & role
# -------------------------
current_user = st.session_state.username
role = USERS.get(current_user, {}).get("role", "viewer")

# Topbar
st.markdown(
    f"""
    <div class="topbar">
      <div style="display:flex; flex-direction:column;">
        <div style="display:flex; gap:12px; align-items:center;">
          <h1 class="main-title">RIC — Terrorism Analytics</h1>
        </div>
        <div class="muted">User: <b>{USERS[current_user]['name']}</b> • Role: <b>{role.upper()}</b></div>
      </div>
      <div style="display:flex; gap:12px; align-items:center;">
        <div class="small">Data file: <b>gti_cleaned.csv</b></div>
        <div class="small">Prepared for reporting & analysis</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Logout (small button)
if st.button("Logout", key="logout_btn"):
    logout()
    st.experimental_rerun()

# -------------------------
# Data loading + upload fallback (in sidebar)
# -------------------------
DATA_FILE = "gti_cleaned.csv"
data = pd.DataFrame()
loaded_from = None

with st.sidebar:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Dataset")
    st.markdown("Upload a CSV to override default dataset (optional).")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
    if uploaded is not None:
        try:
            data = load_csv(io.BytesIO(uploaded.getvalue()))
            loaded_from = "uploaded"
            st.success("Uploaded dataset loaded.")
        except Exception as e:
            st.error(f"Upload failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# default file if not uploaded
if data.empty:
    if Path(DATA_FILE).exists():
        try:
            data = load_csv(DATA_FILE)
            loaded_from = DATA_FILE
        except Exception as e:
            st.error(f"Failed to load {DATA_FILE}: {e}")
            st.stop()
    else:
        st.error(f"❌ Dataset '{DATA_FILE}' not found. Please upload it in the sidebar.")
        st.stop()

# -------------------------
# Normalize & safe conversions
# -------------------------
expected_numeric = ["Incidents", "Fatalities", "Injuries", "Hostages", "Score", "Rank", "Year"]
for c in expected_numeric:
    if c not in data.columns:
        data[c] = 0

if "Country" not in data.columns:
    data["Country"] = np.nan

for c in expected_numeric:
    data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0)

if "iso3c" not in data.columns or data["iso3c"].isnull().any():
    try:
        data["iso3c"] = data["Country"].apply(safe_iso3)
    except Exception:
        data["iso3c"] = None

# -------------------------
# Sidebar: Filters & Controls (dark navy area)
# -------------------------
with st.sidebar:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Filters & Settings")
    years = sorted(list(map(int, data["Year"].dropna().unique()))) if "Year" in data.columns else []
    if years:
        selected_year = st.selectbox("Year", options=["All"] + years, index=0)
        selected_year = None if selected_year == "All" else int(selected_year)
    else:
        selected_year = None

    countries = sorted(data["Country"].dropna().unique().tolist())
    country_options = ["All"] + countries
    selected_country = st.selectbox("Country", country_options, index=0)

    st.markdown("---")
    st.subheader("Visual Options")
    if role == "admin":
        top_n = st.slider("Top N countries", 5, 50, 10, key="top_n")
        show_anim = st.checkbox("Animated map (year)", value=False, key="show_anim")
        show_bubble = st.checkbox("Bubble chart", value=True, key="show_bubble")
        show_heat = st.checkbox("Heatmap", value=True, key="show_heat")
        show_pie = st.checkbox("Show pie (attack types)", value=False, key="show_pie")
    else:
        top_n = st.slider("Top N countries", 5, 50, 10, key="top_n_view", disabled=True)
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
# Apply filters to dataframe
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
# Top metrics row (light cards)
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
    avg_score = float(data["Score"].mean()) if "Score" in data.columns else 0.0
    st.markdown(f"<div class='metric-title'>{avg_score:.2f}</div><div class='metric-sub'>Average Score</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    country_count = int(data["Country"].nunique()) if "Country" in data.columns else 0
    st.markdown(f"<div class='metric-title'>{country_count}</div><div class='metric-sub'>Countries analyzed</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# -------------------------
# Main layout: Map & Visuals
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
                                    animation_frame="Year", color_continuous_scale=px.colors.sequential.Blues, projection="natural earth")
            fig_map.update_layout(margin=dict(t=10, b=0, l=0, r=0), coloraxis_colorbar=dict(title="Score"))
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            map_df = df.groupby(["Country", "iso3c"], as_index=False).agg({"Score": "mean", "Incidents": "sum", "Fatalities": "sum"}) if {"Country", "iso3c"}.issubset(df.columns) else pd.DataFrame()
            if map_df.empty:
                st.info("No map data for selected filters.")
            else:
                fig = px.choropleth(map_df, locations="iso3c", color="Score", hover_name="Country", color_continuous_scale=px.colors.sequential.Blues, projection="natural earth")
                fig.update_layout(margin=dict(t=10, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Map failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Multi-year trend (dual axis)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Incidents vs Fatalities (Multi-year)")
    if "Year" in data.columns:
        try:
            trend = data.groupby("Year").agg({"Incidents": "sum", "Fatalities": "sum", "Score": "mean"}).reset_index()
            fig_tr = go.Figure()
            fig_tr.add_trace(go.Bar(x=trend["Year"], y=trend["Incidents"], name="Incidents", marker_color="#1A73E8"))
            fig_tr.add_trace(go.Scatter(x=trend["Year"], y=trend["Fatalities"], name="Fatalities", yaxis="y2", marker_color="#6C8CFF"))
            fig_tr.update_layout(
                yaxis=dict(title="Incidents"),
                yaxis2=dict(title="Fatalities", overlaying="y", side="right"),
                legend=dict(orientation="h"),
                margin=dict(t=10, b=0),
            )
            st.plotly_chart(fig_tr, use_container_width=True)
        except Exception as e:
            st.error(f"Trend chart failed: {e}")
    else:
        st.info("Year column not available.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Bubble chart
    if show_bubble:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔵 Fatalities vs Injuries (bubble size = Incidents)")
        if {"Fatalities", "Injuries", "Incidents", "Country"}.issubset(data.columns):
            bubble_df = df.groupby("Country", as_index=False).agg({"Fatalities": "sum", "Injuries": "sum", "Incidents": "sum"})
            bubble_df = bubble_df[(bubble_df["Fatalities"] + bubble_df["Injuries"]) > 0]
            if bubble_df.empty:
                st.info("No bubble data for selected filters.")
            else:
                fig_b = px.scatter(bubble_df, x="Injuries", y="Fatalities", size="Incidents", hover_name="Country", size_max=60, color="Incidents", color_continuous_scale=px.colors.sequential.OrRd)
                fig_b.update_layout(margin=dict(t=10, b=0))
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
            fig_bar.update_layout(margin=dict(t=10, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No data for this chart.")
    st.markdown("</div>", unsafe_allow_html=True)

    if show_heat:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔥 Heatmap — Year vs Top Countries (Incidents)")
        if "Year" in data.columns and "Country" in data.columns:
            try:
                top_countries = data.groupby("Country")["Incidents"].sum().nlargest(top_n).index.tolist()
                heat_df = data[data["Country"].isin(top_countries)].pivot_table(values="Incidents", index="Country", columns="Year", aggfunc="sum").fillna(0)
                if heat_df.empty:
                    st.info("No data for heatmap.")
                else:
                    fig_h = px.imshow(heat_df, text_auto=True, aspect="auto", color_continuous_scale=px.colors.sequential.Blues)
                    fig_h.update_layout(margin=dict(t=10, b=0))
                    st.plotly_chart(fig_h, use_container_width=True)
            except Exception as e:
                st.error(f"Heatmap failed: {e}")
        else:
            st.info("Year/Country missing.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Correlation
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔬 Correlation (Numeric Features)")
    numeric_cols = [c for c in ["Incidents", "Fatalities", "Injuries", "Hostages", "Score", "Rank"] if c in data.columns]
    if len(numeric_cols) >= 2:
        corr = data[numeric_cols].corr()
        corr_fig = px.imshow(corr, text_auto=True, color_continuous_scale=px.colors.diverging.RdBu, zmin=-1, zmax=1)
        corr_fig.update_layout(margin=dict(t=10, b=0))
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

def train_rf(n_estimators=150, sample_n=1500):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    sample_n = min(sample_n, len(X))
    if sample_n > 0:
        Xs = X.sample(sample_n, random_state=42)
        ys = y.loc[Xs.index]
        model.fit(Xs, ys)
        return model
    return None

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
        if len(X) > 0:
            with st.spinner("Training model..."):
                model = train_rf(n_estimators=(n_estim if n_estim else 150), sample_n=min(2000, len(X)))
            if model:
                pred = model.predict([[inc, fat, inj, host, rank]])[0]
                st.success(f"Predicted Score: {pred:.2f}")
            else:
                st.warning("Not enough data to train model.")
        else:
            st.warning("No data available to train model.")
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
        if len(X) > 0:
            with st.spinner("Training model..."):
                model = train_rf(n_estimators=150, sample_n=min(1500, len(X)))
            if model:
                pred = model.predict([vals])[0]
                st.success(f"Predicted Score: {pred:.2f}")
            else:
                st.warning("Not enough data to generate prediction.")
        else:
            st.warning("No data available to generate prediction.")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Admin controls & download
# -------------------------
if role == "admin":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("⚙️ Admin Controls")
    try:
        csv_full = data.to_csv(index=False).encode()
        st.download_button("Download full dataset (CSV)", csv_full, file_name="gti_cleaned_full.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Export failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Filtered CSV download (all users)
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("⬇️ Export Filtered Data")
if not df.empty:
    try:
        csv_filtered = df.to_csv(index=False).encode()
        file_name = f"gtd_filtered_{selected_year if selected_year else 'all'}.csv"
        st.download_button("Download filtered CSV", csv_filtered, file_name=file_name, mime="text/csv")
    except Exception as e:
        st.error(f"Filtered export failed: {e}")
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
    try:
        top_sel = df.groupby("Country")["Incidents"].sum().idxmax() if (not df.empty and "Country" in df.columns) else "N/A"
    except Exception:
        top_sel = "N/A"
    st.markdown(f"**Top (selected year)**\n\n{top_sel}")
with ib:
    try:
        spike = int(data.groupby("Year")["Incidents"].sum().idxmax()) if "Year" in data.columns else "N/A"
    except Exception:
        spike = "N/A"
    st.markdown(f"**Year with spike**\n\n{spike}")
with ic:
    try:
        avg_f = float(data["Fatalities"].mean()) if "Fatalities" in data.columns else 0.0
    except Exception:
        avg_f = 0.0
    st.markdown(f"**Avg Fatalities (all years)**\n\n{avg_f:.1f}")
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(f"<div class='footer'>✨ © 2025 Vaishnavi Raut — All rights reserved • Dataset: <i>{loaded_from}</i></div>", unsafe_allow_html=True)
