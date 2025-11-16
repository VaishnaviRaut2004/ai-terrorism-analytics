# step8_ai_dashboard_v3.py
# AI-Powered Terrorism Analytics — Auth + Full Dashboard (Robust hasher)
# Author: for Vaishnavi Raut
# Run: streamlit run step8_ai_dashboard_v3.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import pycountry

# --- Try importing streamlit-authenticator and a compatible Hasher ---
try:
    import streamlit_authenticator as stauth
except Exception:
    st.error("streamlit-authenticator is not installed. Run: pip install streamlit-authenticator")
    st.stop()

# try multiple hasher import paths / APIs to be compatible with different versions
_hasher_fn = None
try:
    # new-ish path in some versions
    from streamlit_authenticator.utilities.hasher import Hasher

    def _hash_pwds(pwds):
        return Hasher(pwds).generate()

    _hasher_fn = _hash_pwds
except Exception:
    try:
        # older API: stauth.Hasher
        def _hash_pwds(pwds):
            return stauth.Hasher(pwds).generate()

        # test it
        _ = _hash_pwds(["test"])
        _hasher_fn = _hash_pwds
    except Exception:
        try:
            # some forks expose lowercase hasher
            def _hash_pwds(pwds):
                return stauth.hasher(pwds).generate()

            _ = _hash_pwds(["test"])
            _hasher_fn = _hash_pwds
        except Exception:
            _hasher_fn = None

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Vaishnavi’s AI Terrorism Analytics", layout="wide", page_icon="🌍")

# -------------------------
# AUTHENTICATION SETUP
# -------------------------
# Users (example). Change/add as needed.
names = ["Vaishnavi Raut", "Demo User"]
usernames = ["vaishnavi", "demo"]
passwords_plain = ["#bharat@123", "demo123"]

if _hasher_fn is None:
    st.error(
        "streamlit-authenticator's hasher utility could not be found in this environment.\n\n"
        "Fix options:\n"
        "• Install a compatible version: pip install streamlit-authenticator\n"
        "• Or run a small script to hash passwords locally and paste hashed values into credentials.\n\n"
        "See the app logs / documentation for more details."
    )
    st.stop()

# Generate hashed passwords (safe each run; you may pre-hash in production)
try:
    hashed_passwords = _hasher_fn(passwords_plain)
except Exception as e:
    st.error(f"Failed to hash passwords: {e}")
    st.stop()

# Build credentials dict expected by Authenticate
credentials = {"usernames": {}}
for uname, display_name, hashed in zip(usernames, names, hashed_passwords):
    credentials["usernames"][uname] = {"name": display_name, "password": hashed}

# Create authenticator
authenticator = stauth.Authenticate(
    credentials,
    cookie_name="vaishnavi_terror_cookie",
    signature_key="vaishnavi_terror_signature_key",
    cookie_expiry_days=1,
)

name, auth_status, username = authenticator.login("Login", "main")

# -------------------------
# If authenticated, show dashboard
# -------------------------
if auth_status:
    authenticator.logout("Logout", "main")
    st.markdown(f"<h3 style='color:#bfc7e6'>Welcome, {name} 🌍</h3>", unsafe_allow_html=True)

    # -------------------------
    # THEME CSS: Black + Blue + Lavender
    # -------------------------
    st.markdown(
        """
        <style>
        :root{
            --bg1: #071022;
            --bg2: #0e1a3a;
            --card: rgba(255,255,255,0.02);
            --lav: #8c92ac;
            --blue: #3a78f2;
            --muted: #95a3b8;
        }
        .stApp { background: linear-gradient(180deg,var(--bg1),var(--bg2)); color:#e9eef8; font-family: Inter, sans-serif; }
        .card { background: var(--card); border-radius:12px; padding:12px; margin-bottom:12px; border:1px solid rgba(255,255,255,0.02); box-shadow: 0 8px 30px rgba(0,0,0,0.6); }
        .metric { padding:10px; border-radius:8px; }
        .small { color:var(--muted); font-size:12px; }
        footer {visibility:hidden}
        #MainMenu {visibility:hidden}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------
    # LOAD DATA (safe)
    # -------------------------
    DATA_FILE = "gti_cleaned.csv"
    if not Path(DATA_FILE).exists():
        st.error("Dataset 'gti_cleaned.csv' not found. Place it next to this script.")
        st.stop()

    @st.cache_data(show_spinner=False)
    def load_data(path):
        df = pd.read_csv(path)
        return df

    data = load_data(DATA_FILE)

    # Normalize column names -> keep as-is but safe checks
    # Ensure Country column exists
    if "Country" not in data.columns and "country" in data.columns:
        data.rename(columns={"country": "Country"}, inplace=True)

    # Robust iso3 fill (do not overwrite valid codes)
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
        data["iso3c"] = data.get("Country", "").apply(safe_iso3_from_name)
    else:
        # strip and fill missing
        data["iso3c"] = data["iso3c"].astype(str).str.strip().replace({"nan": None, "None": None})
        missing_mask = data["iso3c"].isna() | (data["iso3c"] == "")
        if "Country" in data.columns and missing_mask.any():
            data.loc[missing_mask, "iso3c"] = data.loc[missing_mask, "Country"].apply(safe_iso3_from_name)

    # Numeric conversion
    for c in ["Incidents", "Fatalities", "Injuries", "Hostages", "Score", "Rank", "Year"]:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")

    for c in ["Incidents", "Fatalities", "Injuries", "Hostages", "Score", "Rank"]:
        if c in data.columns:
            data[c] = data[c].fillna(0)

    # -------------------------
    # SIDEBAR: filters & options
    # -------------------------
    with st.sidebar:
        st.header("Filters & Settings")
        years = sorted(data["Year"].dropna().astype(int).unique().tolist()) if "Year" in data.columns else []
        selected_year = st.selectbox("Year", years, index=len(years)-1 if years else 0) if years else None

        country_list = ["All"] + (sorted(data["Country"].dropna().unique().tolist()) if "Country" in data.columns else [])
        selected_country = st.selectbox("Country", country_list, index=0)

        st.markdown("---")
        st.subheader("Display Options")
        show_trend = st.checkbox("Show multi-year trend", value=True)
        show_3d = st.checkbox("Enable 3D scatter (may be slower)", value=False)
        st.markdown("---")
        st.subheader("Model")
        retrain = st.checkbox("Retrain full model (slow)", value=False)
        st.markdown("<div class='small'>Tip: retrain only if data or parameters changed.</div>", unsafe_allow_html=True)

    # -------------------------
    # Apply filters
    # -------------------------
    df = data.copy()
    if selected_year is not None and "Year" in df.columns:
        df = df[df["Year"] == int(selected_year)]
    if selected_country and selected_country != "All" and "Country" in df.columns:
        df = df[df["Country"] == selected_country]

    # -------------------------
    # Top metrics
    # -------------------------
    c1, c2, c3, c4 = st.columns([2,1,1,1])
    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        try:
            main_country = data.groupby("Country")["Incidents"].sum().idxmax() if "Country" in data.columns else "N/A"
        except Exception:
            main_country = "N/A"
        st.markdown(f"<div class='metric'><div style='font-size:18px; font-weight:700;'>{main_country}</div><div class='small'>Most affected country (all years)</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        try:
            deadliest_year = int(data.groupby("Year")["Fatalities"].sum().idxmax()) if "Year" in data.columns else 0
        except Exception:
            deadliest_year = 0
        st.markdown(f"<div class='metric'><div style='font-size:18px'>{deadliest_year}</div><div class='small'>Deadliest year</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        avg_score = data["Score"].mean() if "Score" in data.columns else 0
        st.markdown(f"<div class='metric'><div style='font-size:18px'>{avg_score:.2f}</div><div class='small'>Average Score</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        country_count = data["Country"].nunique() if "Country" in data.columns else 0
        st.markdown(f"<div class='metric'><div style='font-size:18px'>{country_count}</div><div class='small'>Countries analyzed</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # -------------------------
    # Map and visuals
    # -------------------------
    left, right = st.columns((2,1))
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader(f"🗺️ Global Terrorism Score — {selected_year if selected_year is not None else 'All Years'}")
        if {"Country","iso3c"}.issubset(df.columns):
            map_df = df.dropna(subset=["iso3c"]).groupby(["Country","iso3c"], as_index=False).agg({"Score":"mean","Fatalities":"sum","Incidents":"sum"})
        else:
            map_df = pd.DataFrame()

        if map_df.empty:
            st.info("No map data (check filters or iso3c values).")
        else:
            fig = px.choropleth(
                map_df,
                locations="iso3c",
                color="Score",
                hover_name="Country",
                hover_data={"Fatalities":True, "Incidents":True, "iso3c":False},
                color_continuous_scale=px.colors.sequential.Plasma,
                projection="natural earth",
            )
            fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if show_trend and "Year" in data.columns:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📈 Multi-Year Trend")
            trend_df = data.groupby("Year").agg({"Incidents":"sum","Fatalities":"sum","Score":"mean"}).reset_index()
            if trend_df.empty:
                st.info("No trend data.")
            else:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=trend_df["Year"], y=trend_df["Incidents"], name="Incidents", mode="lines+markers", line=dict(width=2, color="#3a78f2")))
                fig2.add_trace(go.Scatter(x=trend_df["Year"], y=trend_df["Fatalities"], name="Fatalities", mode="lines+markers", line=dict(width=2, color="#8c92ac")))
                fig2.update_layout(height=360, margin=dict(t=10,b=0,l=0,r=0), legend=dict(orientation="h"), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if show_3d:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("🌐 3D Scatter")
            if len(data) == 0:
                st.info("No data for 3D plot.")
            else:
                sample_n = min(1500, len(data))
                sample = data.sample(sample_n, random_state=42)
                if {"Incidents","Fatalities","Score"}.issubset(sample.columns):
                    fig3 = px.scatter_3d(sample, x="Incidents", y="Fatalities", z="Score", color="Country" if "Country" in sample.columns else None, size="Injuries" if "Injuries" in sample.columns else None, opacity=0.8)
                    fig3.update_layout(height=600, margin=dict(t=10,b=0,l=0,r=0), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("Missing columns for 3D scatter.")
            st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🏆 Top Countries by Fatalities")
        if "Country" in df.columns and "Fatalities" in df.columns:
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
    # Prediction panel
    # -------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🤖 Quick Score Estimator")

    left_p, right_p = st.columns([1,1])
    with left_p:
        i = st.number_input("Incidents", min_value=0, max_value=100000, value=200)
        f = st.number_input("Fatalities", min_value=0, max_value=100000, value=50)
        inj = st.number_input("Injuries", min_value=0, max_value=100000, value=100)
    with right_p:
        h = st.number_input("Hostages", min_value=0, max_value=10000, value=5)
        r = st.slider("Rank", 1, 300, 50)

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
                sample_n = min(3000, len(X_all))
                with st.spinner("Training small model for quick predictions..."):
                    Xs = X_all.sample(sample_n, random_state=42)
                    ys = y_all.loc[Xs.index]
                    model.fit(Xs, ys)
                    model_trained = True
        except Exception as e:
            st.error(f"Model training failed: {e}")

    if st.button("Predict Score"):
        if not model_trained:
            st.error("Model not trained. Toggle retrain or provide more data.")
        else:
            pred = model.predict([[i, f, inj, h, r]])[0]
            st.markdown(f"<div style='font-size:20px; color:#3a78f2; font-weight:600'>🌋 Predicted Score: {pred:.2f}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # footer
    st.markdown("<div style='text-align:center; color:#bfc7e6; padding:10px; font-size:13px;'>✨ © 2025 Vaishnavi Raut — AI-Powered Terrorism Analytics</div>", unsafe_allow_html=True)

# -------------------------
# Auth failure
# -------------------------
elif auth_status is False:
    st.error("❌ Username or password is incorrect")
else:
    st.warning("⚠️ Please enter your username and password")
