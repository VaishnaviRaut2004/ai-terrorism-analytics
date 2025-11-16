# step8_ai_dashboard_v3.py
# AI-Powered Terrorism Analytics — Black + Blue + Lavender Theme
# Author: Vaishnavi Raut
# Run: streamlit run step8_ai_dashboard_v3.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pycountry
from pathlib import Path
import streamlit_authenticator as stauth

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="AI Terrorism Analytics",
    layout="wide",
    page_icon="🌍"
)

# -------------------------
# AUTHENTICATION
# -------------------------
names = ['Vaishnavi Raut', 'Normal User']
usernames = ['vaishnavi', 'user']
passwords = ['#bharat@123', '123']
hashed_passwords = stauth.Hasher(passwords).generate()

credentials = {
    "usernames": {
        usernames[0]: {"name": names[0], "password": hashed_passwords[0]},
        usernames[1]: {"name": names[1], "password": hashed_passwords[1]},
    }
}

authenticator = stauth.Authenticate(
    credentials,
    cookie_name="vaishnavi_terror_cookie",
    signature_key="vaishnavi_terror_signature",
    cookie_expiry_days=1
)

name, auth_status, username = authenticator.login("Login", "main")

if auth_status:
    authenticator.logout("Logout", "main")
    st.markdown(f"<h3 style='color:#8c92ac;'>Welcome, {name} 🌍</h3>", unsafe_allow_html=True)

    # -------------------------
    # PREMIUM BLACK + BLUE + LAVENDER THEME CSS
    # -------------------------
    st.markdown("""
    <style>
    :root{
        --bg1: #0a0f1f;
        --bg2: #111c3b;
        --card: rgba(255,255,255,0.03);
        --glass: rgba(255,255,255,0.035);
        --accent: #8c92ac; /* lavender */
        --blue: #3a78f2;
        --gold: #E6B85A;
        --muted: #9aa6ad;
    }
    .stApp {
        background: linear-gradient(180deg, var(--bg1), var(--bg2));
        color: #e9f0f6;
        font-family: Inter, sans-serif;
    }
    .card {
        background: var(--glass);
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 14px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.6);
        border: 1px solid rgba(255,255,255,0.03);
    }
    .metric { background: linear-gradient(90deg, rgba(58,120,242,0.03), rgba(140,146,172,0.03)); border-radius:10px; padding:10px; color:#fff; }
    .small { color: var(--muted); font-size:12px; }
    .gold { color: var(--gold); font-weight:700; }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

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

    # Ensure iso3c column exists
    def safe_iso3(name):
        try:
            return pycountry.countries.lookup(str(name)).alpha_3
        except:
            return None

    if "iso3c" not in data.columns or data["iso3c"].isnull().any():
        data["iso3c"] = data["Country"].apply(lambda x: safe_iso3(x) if pd.notna(x) else None)

    # Numeric columns
    for c in ["Incidents", "Fatalities", "Injuries", "Hostages", "Score", "Rank", "Year"]:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0)

    # -------------------------
    # SIDEBAR FILTERS
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
        st.markdown("<div class='small'>Tip: Retrain only when data changes.</div>", unsafe_allow_html=True)

    # Filtered data
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
    # MAP + VISUALS
    # -------------------------
    left, right = st.columns((2,1))
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader(f"🗺️ Global Terrorism Score — {selected_year}")
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
            choropleth.update_layout(margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(choropleth, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if show_trend:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📈 Multi-Year Trend")
            trend_df = data.groupby("Year").agg({"Incidents":"sum","Fatalities":"sum","Score":"mean"}).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend_df["Year"], y=trend_df["Incidents"], name="Incidents", mode="lines+markers", line=dict(width=2, color="#3a78f2")))
            fig.add_trace(go.Scatter(x=trend_df["Year"], y=trend_df["Fatalities"], name="Fatalities", mode="lines+markers", line=dict(width=2, color="#8c92ac")))
            fig.update_layout(height=360, margin=dict(t=10,b=0,l=0,r=0), legend=dict(orientation="h"), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if show_3d:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("🌐 3D Scatter — Incidents vs Fatalities vs Score")
            sample = data.sample(min(1500, len(data)), random_state=42)
            fig3 = px.scatter_3d(sample, x="Incidents", y="Fatalities", z="Score",
                                 color="Country", size="Injuries", size_max=18, opacity=0.8)
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
            bar = px.bar(top10.sort_values("Fatalities"), x="Fatalities", y="Country",
                         orientation="h", color="Fatalities",
                         color_continuous_scale=px.colors.sequential.Reds)
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

    X_all = data[["Incidents","Fatalities","Injuries","Hostages","Rank"]].fillna(0)
    y_all = data["Score"].fillna(0)
    model = RandomForestRegressor(n_estimators=150, random_state=42)

    if retrain:
        with st.spinner("Training model on full dataset..."):
            model.fit(X_all, y_all)
            st.success("Model trained on full dataset.")
    else:
        sample_n = min(3000, len(X_all))
        with st.spinner("Training small model for quick predictions..."):
            model.fit(X_all.sample(sample_n, random_state=42), y_all.sample(sample_n, random_state=42))

    if st.button("Predict Score"):
        pred = model.predict([[i, f, inj, h, r]])[0]
        st.markdown(f"<div style='font-size:20px; color:#3a78f2; font-weight:600'>🌋 Predicted Score: {pred:.2f}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # FOOTER
    # -------------------------
    st.markdown("<div style='text-align:center; color:#8c92ac; padding:10px; font-size:13px;'>✨ © 2025 Vaishnavi Raut — AI-Powered Terrorism Analytics</div>", unsafe_allow_html=True)

elif auth_status is False:
    st.error("❌ Username or password is incorrect")
else:
    st.warning("⚠️ Please enter your username and password")
