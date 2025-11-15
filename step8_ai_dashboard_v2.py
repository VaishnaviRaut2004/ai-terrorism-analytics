# step8_ai_dashboard_v2.py
# AI-Powered Terrorism Analytics — Elegant Premium Black + Gold UI
# Author: Redesigned for Vaishnavi Raut
# Run: streamlit run step8_ai_dashboard_v2.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pycountry
from pathlib import Path
import os

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="AI Terrorism Analytics — Premium", layout="wide", page_icon="🌍")

# -------------------------
# ELEGANT PREMIUM CSS (Black + Gold)
# -------------------------
st.markdown(
    """
    <style>
    :root{
      --bg1: #05060a;
      --bg2: #0b1420;
      --card: rgba(255,255,255,0.03);
      --glass: rgba(255,255,255,0.035);
      --gold: #E6B85A;
      --muted: #9aa6ad;
      --accent: linear-gradient(90deg,#0f1724,#13233a);
    }
    .stApp {
      background: linear-gradient(180deg,var(--bg1), var(--bg2));
      color: #e9f0f6;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    /* header */
    .nav {
      padding: 14px 20px;
      border-radius: 12px;
      background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      backdrop-filter: blur(6px);
      margin-bottom: 18px;
      display:flex;
      align-items:center;
      justify-content:space-between;
      box-shadow: 0 6px 30px rgba(0,0,0,0.6);
    }
    .nav h1{ color: var(--gold); margin:0; font-size:20px; letter-spacing:0.2px;}
    .nav .sub{ color: var(--muted); font-size:12px; margin-left:8px;}
    /* cards */
    .card {
      background: var(--glass);
      border-radius: 14px;
      padding: 14px;
      margin-bottom: 14px;
      box-shadow: 0 8px 30px rgba(0,0,0,0.6);
      border: 1px solid rgba(255,255,255,0.03);
      transition: transform .18s ease, box-shadow .2s ease;
    }
    .card:hover { transform: translateY(-6px); box-shadow: 0 18px 40px rgba(0,0,0,0.7); }
    .metric {
      background: linear-gradient(90deg, rgba(255,255,255,0.01), rgba(255,215,102,0.015));
      border-radius: 10px;
      padding: 10px;
      color: #fff;
      text-align: left;
    }
    .small { color: var(--muted); font-size:12px; }
    .gold { color: var(--gold); font-weight:700; }
    .footer { text-align:center; color: #cbbf85; padding:10px; font-size:13px; }
    .muted { color: var(--muted); }
    .chip { background: rgba(255,255,255,0.02); padding:6px 10px; border-radius:999px; color:var(--muted); font-size:12px; border:1px solid rgba(255,255,255,0.02); }
    /* hide default hamburger */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# HEADER
# -------------------------
st.markdown(
    f"""
    <div class="nav">
      <div>
        <h1>🌍 AI-Powered Terrorism Analytics</h1>
        <div class="sub">Elegant • Interactive • Insight-driven</div>
      </div>
      <div style="display:flex; gap:10px; align-items:center;">
        <div class="chip">Data: gti_cleaned.csv</div>
        <div class="chip">Theme: Premium Black+Gold</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# LOAD DATA
# -------------------------
DATA_FILE = "gti_cleaned.csv"
if not Path(DATA_FILE).exists():
    st.error("❌ Dataset not found. Place 'gti_cleaned.csv' in the same folder as this app and restart.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path)
    return df

data = load_data(DATA_FILE)

# expected columns check
expected_cols = {"iso3c", "Country", "Rank", "Score", "Incidents", "Fatalities", "Injuries", "Hostages", "Year"}
missing = expected_cols - set(data.columns)
if missing:
    st.warning(f"⚠️ Missing expected columns: {', '.join(missing)} — try renaming columns to match or provide a clean CSV.")
# fill iso3 if missing or broken
def safe_iso3(name):
    try:
        return pycountry.countries.lookup(str(name)).alpha_3
    except:
        return None

if "iso3c" not in data.columns or data["iso3c"].isnull().any():
    data["iso3c"] = data["Country"].apply(lambda x: safe_iso3(x) if pd.notna(x) else None)

# convert numeric columns
for c in ["Incidents", "Fatalities", "Injuries", "Hostages", "Score", "Rank", "Year"]:
    if c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0)

# -------------------------
# SIDEBAR CONTROLS
# -------------------------
with st.sidebar:
    st.header("Filters & Settings")
    years = sorted(data["Year"].unique().astype(int).tolist())
    selected_year = st.selectbox("Year", years, index=len(years)-1)
    country_list = ["All"] + sorted(data["Country"].unique().tolist())
    selected_country = st.selectbox("Country", country_list, index=0)
    st.markdown("---")
    st.subheader("Display Options")
    show_trend = st.checkbox("Show multi-year trend chart", value=True)
    show_3d = st.checkbox("Enable 3D scatter (may be slower)", value=False)
    st.markdown("---")
    st.subheader("Model")
    retrain = st.checkbox("Retrain model fully (slow)", value=False)
    st.markdown("<div class='small'>Tip: Retrain only when data or parameters change.</div>", unsafe_allow_html=True)

# filtered df
df = data.copy()
df = df[df["Year"] == int(selected_year)]
if selected_country != "All":
    df = df[df["Country"] == selected_country]

# -------------------------
# TOP METRICS
# -------------------------
c1, c2, c3, c4 = st.columns([2,1,1,1])
with c1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    main_country = data.groupby("Country")["Incidents"].sum().idxmax() if not data.empty else "N/A"
    st.markdown(f"<div class='metric'><div class='gold' style='font-size:18px'>🌋 {main_country}</div><div class='small'>Most affected country (all years)</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    deadliest_year = int(data.groupby("Year")["Fatalities"].sum().idxmax()) if not data.empty else 0
    st.markdown(f"<div class='metric'><div style='font-size:18px'>{deadliest_year}</div><div class='small'>Deadliest year</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    avg_score = data["Score"].mean()
    st.markdown(f"<div class='metric'><div style='font-size:18px'>{avg_score:.2f}</div><div class='small'>Average Score</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with c4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    country_count = data["Country"].nunique()
    st.markdown(f"<div class='metric'><div style='font-size:18px'>{country_count}</div><div class='small'>Countries analyzed</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# -------------------------
# MAP + VISUALS LAYOUT
# -------------------------
left, right = st.columns((2,1))
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(f"🗺️ Global Terrorism Score — {selected_year}")
    # aggregate for map
    map_df = df.groupby(["Country","iso3c"], as_index=False).agg({"Score":"mean","Fatalities":"sum","Incidents":"sum"})
    if map_df.empty:
        st.info("No data for selected filters")
    else:
        choropleth = px.choropleth(
            map_df,
            locations="iso3c",
            color="Score",
            hover_name="Country",
            hover_data={"Fatalities":True,"Incidents":True,"iso3c":False},
            color_continuous_scale=px.colors.sequential.Plasma,
            projection="natural earth"
        )
        choropleth.update_layout(margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_colorbar=dict(title="Score"))
        st.plotly_chart(choropleth, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if show_trend:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📈 Multi-Year Trend")
        trend_df = data.groupby("Year").agg({"Incidents":"sum","Fatalities":"sum","Score":"mean"}).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend_df["Year"], y=trend_df["Incidents"], name="Incidents", mode="lines+markers", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=trend_df["Year"], y=trend_df["Fatalities"], name="Fatalities", mode="lines+markers", line=dict(width=2)))
        fig.update_layout(height=360, margin=dict(t=10,b=0,l=0,r=0), legend=dict(orientation="h"), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if show_3d:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🌐 3D Scatter — Incidents vs Fatalities vs Score")
        # 3D using a sample to keep responsive
        sample = data.sample(min(1500, len(data)), random_state=42)
        fig3 = px.scatter_3d(sample, x="Incidents", y="Fatalities", z="Score", color="Country", size="Injuries", size_max=18, opacity=0.8)
        fig3.update_layout(height=600, margin=dict(t=10,b=0,l=0,r=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏆 Top Countries by Fatalities")
    top10 = df.groupby("Country")["Fatalities"].sum().nlargest(10).reset_index()
    if top10.empty:
        st.info("No data for selected filters.")
    else:
        bar = px.bar(top10.sort_values("Fatalities"), x="Fatalities", y="Country", orientation="h", color="Fatalities", color_continuous_scale=px.colors.sequential.Reds)
        bar.update_layout(margin=dict(t=0,b=0,l=0,r=0), height=360, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔬 Correlation")
    corr = data[["Incidents","Fatalities","Injuries","Hostages","Score","Rank"]].corr()
    corr_fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale=px.colors.diverging.RdBu)
    corr_fig.update_layout(height=320, margin=dict(t=0,b=0,l=0,r=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(corr_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# PREDICTION PANEL
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🤖 Score Estimator — Quick Prediction")

left_p, right_p = st.columns([1,1])
with left_p:
    i = st.number_input("Incidents", min_value=0, max_value=100000, value=200)
    f = st.number_input("Fatalities", min_value=0, max_value=100000, value=50)
    inj = st.number_input("Injuries", min_value=0, max_value=100000, value=100)
with right_p:
    h = st.number_input("Hostages", min_value=0, max_value=10000, value=5)
    r = st.slider("Rank", 1, 300, 50)

# prepare model training (safe & responsive)
X_all = data[["Incidents","Fatalities","Injuries","Hostages","Rank"]].fillna(0)
y_all = data["Score"].fillna(0)
model = RandomForestRegressor(n_estimators=150, random_state=42)
if retrain:
    with st.spinner("Training model on full dataset..."):
        model.fit(X_all, y_all)
        st.success("Model trained on full dataset.")
else:
    # quick train on sample for responsiveness
    sample_n = min(3000, len(X_all))
    with st.spinner("Training small model for quick predictions..."):
        model.fit(X_all.sample(sample_n, random_state=42), y_all.sample(sample_n, random_state=42))

if st.button("Predict Score"):
    pred = model.predict([[i, f, inj, h, r]])[0]
    st.markdown(f"<div style='font-size:20px; color:var(--gold); font-weight:600'>🌋 Predicted Score: {pred:.2f}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# AUTO INSIGHTS
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧠 Auto Insights")

col_a, col_b, col_c = st.columns(3)
with col_a:
    top_country_year = df.groupby("Country")["Incidents"].sum().idxmax() if not df.empty else "N/A"
    st.markdown(f"**Top (selected year)**\n\n{top_country_year}")
with col_b:
    spike = int(data.groupby("Year")["Incidents"].sum().idxmax()) if not data.empty else 0
    st.markdown(f"**Year with spike**\n\n{spike}")
with col_c:
    avg_fatal = data["Fatalities"].mean()
    st.markdown(f"**Avg Fatalities (all years)**\n\n{avg_fatal:.1f}")

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# FOOTER
# -------------------------
st.markdown("<div class='footer'>✨ © 2025 Vaishnavi Raut — AI-Powered Terrorism Analytics (Premium)</div>", unsafe_allow_html=True)
