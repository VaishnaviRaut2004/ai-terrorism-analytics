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
import pycountry
from pathlib import Path

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
present = set(data.columns)
missing = expected_cols - present
if missing:
    # only warn — some columns may be legitimately absent but handled later
    st.warning(f"⚠️ Missing expected columns: {', '.join(missing)} — try renaming columns to match or provide a clean CSV.")

# -------------------------
# ISO3 handling (robust)
# -------------------------
# helper: try several lookups and safe cleaning
def safe_iso3_from_name(name):
    if pd.isna(name):
        return None
    n = str(name).strip()
    if not n:
        return None
    # direct pass-through if already a 3-letter code
    if len(n) == 3 and n.isalpha():
        return n.upper()
    try:
        c = pycountry.countries.lookup(n)
        return getattr(c, 'alpha_3', None)
    except Exception:
        # try fuzzy by common name casing
        try:
            c = pycountry.countries.get(name=n)
            if c:
                return getattr(c, 'alpha_3', None)
        except Exception:
            return None

# only create or fill missing iso3c values, don't overwrite good data
if 'iso3c' not in data.columns:
    data['iso3c'] = data.get('Country', '').apply(safe_iso3_from_name)
else:
    # strip and keep existing where present; fill nulls using Country
    data['iso3c'] = data['iso3c'].astype(str).str.strip().replace({'nan': None, 'None': None})
    mask_missing = data['iso3c'].isna() | (data['iso3c'] == '')
    if 'Country' in data.columns and mask_missing.any():
        data.loc[mask_missing, 'iso3c'] = data.loc[mask_missing, 'Country'].apply(safe_iso3_from_name)

# -------------------------
# Clean & convert numeric columns safely
# -------------------------
numeric_cols = ["Incidents", "Fatalities", "Injuries", "Hostages", "Score", "Rank", "Year"]
for c in numeric_cols:
    if c in data.columns:
        data[c] = pd.to_numeric(data[c], errors='coerce')

# replace NaN numeric with 0 only for columns that we fill; keep Year NaN for filtering purposes
for c in ["Incidents", "Fatalities", "Injuries", "Hostages", "Score", "Rank"]:
    if c in data.columns:
        data[c] = data[c].fillna(0)

# -------------------------
# SIDEBAR CONTROLS
# -------------------------
with st.sidebar:
    st.header("Filters & Settings")
    # years: dropna and cast to int where possible
    if 'Year' in data.columns:
        years = sorted(data['Year'].dropna().astype(int).unique().tolist())
    else:
        years = []
    if years:
        selected_year = st.selectbox("Year", years, index=len(years)-1)
    else:
        selected_year = None

    # countries
    if 'Country' in data.columns:
        country_list = ["All"] + sorted(data['Country'].dropna().unique().tolist())
    else:
        country_list = ["All"]
    selected_country = st.selectbox("Country", country_list, index=0)

    st.markdown("---")
    st.subheader("Display Options")
    show_trend = st.checkbox("Show multi-year trend chart", value=True)
    show_3d = st.checkbox("Enable 3D scatter (may be slower)", value=False)
    st.markdown("---")
    st.subheader("Model")
    retrain = st.checkbox("Retrain model fully (slow)", value=False)
    st.markdown("<div class='small'>Tip: Retrain only when data or parameters change.</div>", unsafe_allow_html=True)

# -------------------------
# FILTERED DF (safe)
# -------------------------
df = data.copy()
if selected_year is not None and 'Year' in df.columns:
    df = df[df['Year'] == int(selected_year)]
if selected_country and selected_country != 'All' and 'Country' in df.columns:
    df = df[df['Country'] == selected_country]

# -------------------------
# TOP METRICS (safe guards)
# -------------------------
c1, c2, c3, c4 = st.columns([2,1,1,1])
with c1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    try:
        main_country = data.groupby("Country")["Incidents"].sum().idxmax() if 'Country' in data.columns and not data.empty else "N/A"
    except Exception:
        main_country = "N/A"
    st.markdown(f"<div class='metric'><div class='gold' style='font-size:18px'>🌋 {main_country}</div><div class='small'>Most affected country (all years)</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    try:
        deadliest_year = int(data.groupby("Year")["Fatalities"].sum().idxmax()) if 'Year' in data.columns and not data.empty else 0
    except Exception:
        deadliest_year = 0
    st.markdown(f"<div class='metric'><div style='font-size:18px'>{deadliest_year}</div><div class='small'>Deadliest year</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    avg_score = data['Score'].mean() if 'Score' in data.columns else 0
    st.markdown(f"<div class='metric'><div style='font-size:18px'>{avg_score:.2f}</div><div class='small'>Average Score</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with c4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    country_count = data['Country'].nunique() if 'Country' in data.columns else 0
    st.markdown(f"<div class='metric'><div style='font-size:18px'>{country_count}</div><div class='small'>Countries analyzed</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# -------------------------
# MAP + VISUALS LAYOUT
# -------------------------
left, right = st.columns((2,1))
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(f"🗺️ Global Terrorism Score — {selected_year if selected_year is not None else 'All Years'}")
    # aggregate for map — drop rows without iso3c
    if {'Country','iso3c'}.issubset(df.columns):
        map_df = df.dropna(subset=['iso3c']).groupby(["Country","iso3c"], as_index=False).agg({"Score":"mean","Fatalities":"sum","Incidents":"sum"})
    else:
        map_df = pd.DataFrame()

    if map_df.empty:
        st.info("No data for selected filters or missing iso3c codes. Try adding/filling iso3c column.")
    else:
        choropleth = px.choropleth(
            map_df,
            locations="iso3c",
            color="Score",
            hover_name="Country",
            hover_data={"Fatalities":True,"Incidents":True,"iso3c":False},
            color_continuous_scale=px.colors.sequential.Plasma,
            projection="natural earth",
        )
        choropleth.update_layout(margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_colorbar=dict(title="Score"))
        st.plotly_chart(choropleth, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if show_trend and 'Year' in data.columns:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📈 Multi-Year Trend")
        trend_df = data.groupby("Year").agg({"Incidents":"sum","Fatalities":"sum","Score":"mean"}).reset_index()
        if trend_df.empty:
            st.info("No year-wise data available.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend_df["Year"], y=trend_df["Incidents"], name="Incidents", mode="lines+markers", line=dict(width=2)))
            fig.add_trace(go.Scatter(x=trend_df["Year"], y=trend_df["Fatalities"], name="Fatalities", mode="lines+markers", line=dict(width=2)))
            fig.update_layout(height=360, margin=dict(t=10,b=0,l=0,r=0), legend=dict(orientation="h"), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if show_3d:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🌐 3D Scatter — Incidents vs Fatalities vs Score")
        if data.empty:
            st.info("No data to plot 3D scatter.")
        else:
            sample_n = min(1500, len(data))
            sample = data.sample(sample_n, random_state=42)
            # ensure the columns exist
            if {'Incidents','Fatalities','Score'}.issubset(sample.columns):
                fig3 = px.scatter_3d(sample, x="Incidents", y="Fatalities", z="Score", color="Country" if 'Country' in sample.columns else None, size="Injuries" if 'Injuries' in sample.columns else None, size_max=18, opacity=0.8)
                fig3.update_layout(height=600, margin=dict(t=10,b=0,l=0,r=0), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Required columns for 3D plot are missing.")
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏆 Top Countries by Fatalities")
    if 'Country' in df.columns and 'Fatalities' in df.columns:
        top10 = df.groupby("Country")["Fatalities"].sum().nlargest(10).reset_index()
    else:
        top10 = pd.DataFrame()

    if top10.empty:
        st.info("No data for selected filters.")
    else:
        bar = px.bar(top10.sort_values("Fatalities"), x="Fatalities", y="Country", orientation="h", color="Fatalities", color_continuous_scale=px.colors.sequential.Reds)
        bar.update_layout(margin=dict(t=0,b=0,l=0,r=0), height=360, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔬 Correlation")
    corr_cols = [c for c in ["Incidents","Fatalities","Injuries","Hostages","Score","Rank"] if c in data.columns]
    if len(corr_cols) >= 2:
        corr = data[corr_cols].corr()
        corr_fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale=px.colors.diverging.RdBu)
        corr_fig.update_layout(height=320, margin=dict(t=0,b=0,l=0,r=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(corr_fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns to compute correlation.")
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
feature_cols = ["Incidents","Fatalities","Injuries","Hostages","Rank"]
X_all = data.reindex(columns=feature_cols).fillna(0)
# ensure y aligns with X_all length
if 'Score' in data.columns:
    y_all = data['Score'].fillna(0)
else:
    y_all = pd.Series(np.zeros(len(X_all)))

model = RandomForestRegressor(n_estimators=150, random_state=42)
model_trained = False

if len(X_all) == 0 or len(y_all) == 0:
    st.warning("Not enough data to train prediction model.")
else:
    if retrain:
        with st.spinner("Training model on full dataset..."):
            try:
                model.fit(X_all, y_all)
                model_trained = True
                st.success("Model trained on full dataset.")
            except Exception as e:
                st.error(f"Model training failed: {e}")
    else:
        # quick train on sample for responsiveness
        sample_n = min(3000, len(X_all))
        if sample_n > 0:
            with st.spinner("Training small model for quick predictions..."):
                try:
                    Xs = X_all.sample(sample_n, random_state=42)
                    ys = y_all.loc[Xs.index]
                    model.fit(Xs, ys)
                    model_trained = True
                except Exception as e:
                    st.error(f"Quick model training failed: {e}")

if st.button("Predict Score"):
    if not model_trained:
        st.error("Model is not trained. Please enable retrain or provide more data.")
    else:
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
    try:
        top_country_year = df.groupby("Country")["Incidents"].sum().idxmax() if 'Country' in df.columns and not df.empty else "N/A"
    except Exception:
        top_country_year = "N/A"
    st.markdown(f"**Top (selected year)**\n\n{top_country_year}")
with col_b:
    try:
        spike = int(data.groupby("Year")["Incidents"].sum().idxmax()) if 'Year' in data.columns and not data.empty else 0
    except Exception:
        spike = 0
    st.markdown(f"**Year with spike**\n\n{spike}")
with col_c:
    avg_fatal = data['Fatalities'].mean() if 'Fatalities' in data.columns else 0
    st.markdown(f"**Avg Fatalities (all years)**\n\n{avg_fatal:.1f}")

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# FOOTER
# -------------------------
st.markdown("<div class='footer'>✨ © 2025 Vaishnavi Raut — AI-Powered Terrorism Analytics (Premium)</div>", unsafe_allow_html=True)
