# step8_ai_dashboard_v2.py
# AI-Powered Terrorism Analytics — Emerald Matrix Theme (Green + Teal + White)
# PIN Auth with Roles: Admin (edit) / Viewer (read-only)
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
st.set_page_config(page_title="Emerald Matrix — Terrorism Analytics", layout="wide", page_icon="🟩")

# -------------------------
# USERS with ROLES
# -------------------------
USERS = {
    "vaishnavi": {"name": "Vaishnavi Raut", "pin": "1981", "role": "admin"},
    "viewer": {"name": "Read Only", "pin": "0000", "role": "viewer"}
}

# -------------------------
# SESSION-STATE INIT
# -------------------------
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False
if "username" not in st.session_state:
    st.session_state.username = None
if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0

# -------------------------
# LOGIN WIDGET (PIN)
# -------------------------
def login_widget():
    st.markdown(
        """
        <div style="display:flex; gap:12px; align-items:center;">
          <h2 style="margin:0; color:#0b8043;">🔐 Sign in</h2>
          <div style="color:#27664b; margin-left:6px; font-size:13px;">(username & PIN)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    u = st.text_input("Username", key="login_user_em", placeholder="vaishnavi")
    p = st.text_input("PIN", key="login_pin_em", type="password", placeholder="1981")
    if st.button("Login", key="login_btn_em"):
        st.session_state.login_attempts += 1
        if u in USERS and p == USERS[u]["pin"]:
            st.session_state.auth_ok = True
            st.session_state.username = u
            st.success(f"Welcome, {USERS[u]['name']} ({USERS[u]['role'].upper()})")
        else:
            st.error("Invalid username or PIN.")
            if st.session_state.login_attempts >= 5:
                st.warning("Too many failed attempts. Refresh to try again.")

def logout():
    st.session_state.auth_ok = False
    st.session_state.username = None

# -------------------------
# SHOW LOGIN IF NOT AUTHENTICATED
# -------------------------
if not st.session_state.auth_ok:
    st.markdown(
        """
        <style>
        .login-wrap { max-width:880px; margin:28px auto; padding:18px; border-radius:12px;
            background: linear-gradient(180deg,#ffffff,#f7fffa); box-shadow: 0 8px 30px rgba(3,60,37,0.06);
            border:1px solid rgba(3,60,37,0.06);}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='login-wrap'>", unsafe_allow_html=True)
    st.markdown("<h1 style='color:#0b8043; margin:0'>🟩 Emerald Matrix — AI Terrorism Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#27664b; margin-top:6px;'>Secure dashboard — sign in to continue (admin / viewer).</p>", unsafe_allow_html=True)
    login_widget()
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# -------------------------
# After login: determine role
# -------------------------
current_user = st.session_state.username
role = USERS.get(current_user, {}).get("role", "viewer")

# -------------------------
# THEME CSS — Emerald Matrix (Green + Teal + White)
# -------------------------
st.markdown(
    """
    <style>
    :root{
        --bg1:#f6fffb;
        --bg2:#e8fff5;
        --emerald:#0b8043;
        --teal:#00796b;
        --muted:#2f5f54;
        --card: rgba(255,255,255,0.96);
    }
    .stApp { background: linear-gradient(180deg,var(--bg1),var(--bg2)); color:#05331f; font-family: Inter, ui-sans-serif, system-ui; }
    .topbar { padding:12px 18px; margin-bottom:14px; border-radius:12px; display:flex; justify-content:space-between; align-items:center;
              background: linear-gradient(90deg, rgba(255,255,255,0.85), rgba(255,255,255,0.9)); box-shadow: 0 8px 30px rgba(3,60,37,0.04); border:1px solid rgba(3,60,37,0.03);}
    .card { background: var(--card); border-radius:12px; padding:14px; margin-bottom:12px; box-shadow: 0 8px 24px rgba(3,60,37,0.03); border:1px solid rgba(3,60,37,0.03);}
    .metric { padding:10px; border-radius:10px; }
    .small { color:var(--muted); font-size:13px; }
    .chip { background: linear-gradient(90deg,#fff,#f1fff6); padding:6px 10px; border-radius:999px; color:var(--muted); border:1px solid rgba(3,60,37,0.03); }
    footer { text-align:center; color:var(--muted); padding:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# HEADER TOPBAR
# -------------------------
st.markdown(
    f"""
    <div class="topbar">
      <div>
        <h2 style="margin:0; color:#0b8043;">🟩 Emerald Matrix — AI Terrorism Analytics</h2>
        <div class="small">User: <b>{USERS[current_user]['name']}</b> • Role: <b>{role.upper()}</b></div>
      </div>
      <div style="display:flex; gap:10px; align-items:center;">
        <div class="chip">Data: gti_cleaned.csv</div>
        <div class="chip">Theme: Emerald Matrix</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Logout button
if st.button("Logout", key="logout_em"):
    logout()
    st.experimental_rerun()

# -------------------------
# LOAD DATA
# -------------------------
DATA_FILE = "gti_cleaned.csv"
if not Path(DATA_FILE).exists():
    st.error("❌ Dataset 'gti_cleaned.csv' not found. Please add it.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_data(path):
    return pd.read_csv(path)

data = load_data(DATA_FILE)

# -------------------------
# Clean - ISO3 column
# -------------------------
def safe_iso3(name):
    try:
        if pd.isna(name):
            return None
        s = str(name).strip()
        if len(s) == 3 and s.isalpha():
            return s.upper()
        return pycountry.countries.lookup(s).alpha_3
    except:
        return None

if "iso3c" not in data.columns:
    if "Country" in data.columns:
        data["iso3c"] = data["Country"].apply(safe_iso3)
    else:
        data["iso3c"] = None

# -------------------------
# Correct numeric conversion (FINAL correct version)
# -------------------------
for c in ["Incidents", "Fatalities", "Injuries", "Hostages", "Score", "Rank", "Year"]:
    if c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")

for c in ["Incidents", "Fatalities", "Injuries", "Hostages", "Score", "Rank"]:
    if c in data.columns:
        data[c] = data[c].fillna(0)

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.header("Filters & Settings")
    years = sorted(data["Year"].dropna().astype(int).unique().tolist())
    selected_year = st.selectbox("Year", years, index=len(years)-1)

    country_list = ["All"] + sorted(data["Country"].dropna().unique().tolist())
    selected_country = st.selectbox("Country", country_list, index=0)

    st.markdown("---")
    st.subheader("Visualization Options")

    top_n = st.slider("Top N countries", 5, 30, 10)

    show_anim_map = st.checkbox("Animated map (year)", value=True)
    show_heatmap = st.checkbox("Heatmap Year vs Countries", value=True)
    show_bubble = st.checkbox("Bubble Chart", value=True)
    show_pie = st.checkbox("Top-country pie", value=True)

    st.markdown("---")
    st.subheader("Model")

    retrain = False
    n_estim = 150

    if role == "admin":
        retrain = st.checkbox("Retrain model", value=False)
        n_estim = st.number_input("RF n_estimators", min_value=10, max_value=500, value=150)

# -------------------------
# Apply filters
# -------------------------
df = data.copy()
df = df[df["Year"] == selected_year]
if selected_country != "All":
    df = df[df["Country"] == selected_country]

# -------------------------
# TOP METRICS
# -------------------------
c1, c2, c3, c4 = st.columns([2,1,1,1])

with c1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    top_country = data.groupby("Country")["Incidents"].sum().idxmax()
    st.markdown(f"<b>{top_country}</b><br><span class='small'>Most affected country</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    deadliest_year = int(data.groupby("Year")["Fatalities"].sum().idxmax())
    st.markdown(f"<b>{deadliest_year}</b><br><span class='small'>Deadliest year</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    avg_score = data["Score"].mean()
    st.markdown(f"<b>{avg_score:.2f}</b><br><span class='small'>Average Score</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    ccount = data["Country"].nunique()
    st.markdown(f"<b>{ccount}</b><br><span class='small'>Countries analyzed</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------
# MAP & CHARTS
# -------------------------
left, right = st.columns((2,1))

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(f"🗺 Global Terrorism Score — {selected_year}")
    try:
        map_df = df.groupby(["Country","iso3c"], as_index=False).agg({
            "Score":"mean","Incidents":"sum","Fatalities":"sum"
        })
        fig = px.choropleth(
            map_df, locations="iso3c", color="Score", hover_name="Country",
            color_continuous_scale=px.colors.sequential.Tealgrn
        )
        fig.update_layout(margin=dict(t=5,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Map data missing.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Trend
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Multi-Year Trend")
    trend = data.groupby("Year").agg({"Incidents":"sum","Fatalities":"sum"}).reset_index()
    figT = go.Figure()
    figT.add_trace(go.Bar(x=trend["Year"], y=trend["Incidents"], name="Incidents", marker_color="#66bb6a"))
    figT.add_trace(go.Scatter(x=trend["Year"], y=trend["Fatalities"], name="Fatalities", yaxis="y2"))
    figT.update_layout(yaxis=dict(title="Incidents"),
                       yaxis2=dict(title="Fatalities", overlaying="y", side="right"))
    st.plotly_chart(figT, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    # Top N
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏆 Top Countries by Fatalities")
    top10 = df.groupby("Country")["Fatalities"].sum().nlargest(top_n).reset_index()
    bar = px.bar(top10, x="Fatalities", y="Country", orientation="h",
                 color="Fatalities", color_continuous_scale=px.colors.sequential.Teal)
    st.plotly_chart(bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Heatmap
    if show_heatmap:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔥 Heatmap")
        top_countries = data.groupby("Country")["Incidents"].sum().nlargest(top_n).index
        heat_df = data[data["Country"].isin(top_countries)].pivot_table(
            values="Incidents", index="Country", columns="Year", aggfunc="sum"
        ).fillna(0)
        figH = px.imshow(heat_df, text_auto=True, color_continuous_scale=px.colors.sequential.Mint)
        st.plotly_chart(figH, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# PREDICTION PANEL
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🤖 Score Predictor")

feature_cols = ["Incidents","Fatalities","Injuries","Hostages","Rank"]
for c in feature_cols:
    if c not in data.columns:
        data[c] = 0

X = data[feature_cols].fillna(0)
y = data["Score"].fillna(0)

if role == "admin":
    colA, colB = st.columns(2)
    with colA:
        inc = st.number_input("Incidents", 0, 100000, 200)
        fat = st.number_input("Fatalities", 0, 100000, 50)
        inj = st.number_input("Injuries", 0, 100000, 100)
    with colB:
        host = st.number_input("Hostages", 0, 10000, 5)
        rank = st.slider("Rank", 1, 300, 50)

    if st.button("Predict (Admin)"):
        model = RandomForestRegressor(n_estimators=n_estim, random_state=42)
        sample_n = min(2000, len(X))
        Xs = X.sample(sample_n, random_state=42)
        ys = y.loc[Xs.index]
        model.fit(Xs, ys)
        pred = model.predict([[inc, fat, inj, host, rank]])[0]
        st.success(f"Predicted Score: {pred:.2f}")

else:
    scenarios = {
        "Low": [10,0,1,0,100],
        "Medium": [500,20,50,2,60],
        "High": [5000,300,800,20,10]
    }
    sel = st.selectbox("Scenario", scenarios.keys())
    if st.button("Predict (Viewer)"):
        vals = scenarios[sel]
        model = RandomForestRegressor(n_estimators=150, random_state=42)
        Xs = X.sample(min(1500,len(X)), random_state=42)
        ys = y.loc[Xs.index]
        model.fit(Xs, ys)
        pred = model.predict([vals])[0]
        st.success(f"Predicted Score: {pred:.2f}")

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# FOOTER
# -------------------------
st.markdown("<div style='text-align:center; color:#27664b; padding:10px;'>✨ © 2025 Vaishnavi Raut — Emerald Matrix Dashboard</div>", unsafe_allow_html=True)
