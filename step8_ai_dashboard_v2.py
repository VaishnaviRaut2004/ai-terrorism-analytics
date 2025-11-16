import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import pycountry
from pathlib import Path
import streamlit_authenticator as stauth

# Page config
st.set_page_config(page_title="AI Terrorism Analytics — Premium", layout="wide", page_icon="🌍")

# Premium Black + Gold CSS styling
st.markdown("""
<style>
:root {
  --bg1: #05060a;
  --bg2: #0b1420;
  --card: rgba(255, 255, 255, 0.03);
  --glass: rgba(255, 255, 255, 0.035);
  --gold: #E6B85A;
  --muted: #9aa6ad;
  --accent: linear-gradient(90deg, #0f1724, #13233a);
}
.stApp {
  background: linear-gradient(180deg, var(--bg1), var(--bg2));
  color: #e9f0f6;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}
.nav {
  padding: 14px 20px;
  border-radius: 12px;
  background: var(--glass);
  backdrop-filter: blur(6px);
  margin-bottom: 18px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 6px 30px rgba(0, 0, 0, 0.6);
}
.nav h1 {
  color: var(--gold);
  margin: 0;
  font-size: 20px;
  letter-spacing: 0.2px;
}
.nav .sub {
  color: var(--muted);
  font-size: 12px;
  margin-left: 8px;
}
.card {
  background: var(--glass);
  border-radius: 14px;
  padding: 14px;
  margin-bottom: 14px;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.6);
  border: 1px solid rgba(255, 255, 255, 0.03);
  transition: transform .18s ease, box-shadow .2s ease;
}
.card:hover {
  transform: translateY(-6px);
  box-shadow: 0 18px 40px rgba(0, 0, 0, 0.7);
}
.metric {
  background: linear-gradient(90deg, rgba(255,255,255,0.01), rgba(255,215,102,0.015));
  border-radius: 10px;
  padding: 10px;
  color: #fff;
  text-align: left;
}
.small {
  color: var(--muted);
  font-size: 12px;
}
.gold {
  color: var(--gold);
  font-weight: 700;
}
.footer {
  text-align: center;
  color: #cbbf85;
  padding: 10px;
  font-size: 13px;
}
.muted {
  color: var(--muted);
}
.chip {
  background: rgba(255, 255, 255, 0.02);
  padding: 6px 10px;
  border-radius: 999px;
  color: var(--muted);
  font-size: 12px;
  border: 1px solid rgba(255, 255, 255, 0.02);
}
#MainMenu {
  visibility: hidden;
}
footer {
  visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# User credentials
users = {
    "admin": {"name": "Administrator", "password": "adminpassword", "role": "admin"},
    "user": {"name": "User", "password": "userpassword", "role": "user"},
}
hashed_passwords = stauth.Hasher([users[u]["password"] for u in users]).generate()
authenticator = stauth.Authenticate(
    list(users.keys()),
    [users[u]["name"] for u in users],
    hashed_passwords,
    "cookie_name",
    "signature_key",
    cookie_expiry_days=1
)

name, auth_status, username = authenticator.login("Login", "main")

if not auth_status:
    if auth_status is False:
        st.error("Username/password incorrect")
    else:
        st.info("Please enter your credentials")
    st.stop()

role = users[username]["role"]

# Header UI
st.markdown(f"""
<div class="nav">
  <div>
    <h1>🌍 AI-Powered Terrorism Analytics</h1>
    <div class="sub">Elegant • Interactive • Insight-driven</div>
  </div>
  <div style="display:flex; gap:10px; align-items:center;">
    <div class="chip">Data: gti_cleaned.csv</div>
    <div class="chip">Theme: Premium Black+Gold</div>
    <div class="chip">Role: {role.title()}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Load dataset
DATA_FILE = "gti_cleaned.csv"
if not Path(DATA_FILE).exists():
    st.error("❌ Dataset not found. Place 'gti_cleaned.csv' in the same folder and restart.")
    st.stop()

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

data = load_data(DATA_FILE)

# ISO3c helper and cleanup omitted for brevity (same as before)

# Filter and numeric cleanup omitted for brevity (same as before)

# Sidebar filters for year & country common for both roles
with st.sidebar:
    st.header("Filters & Settings")
    years = sorted(data['Year'].dropna().astype(int).unique().tolist()) if 'Year' in data.columns else []
    selected_year = st.selectbox("Year", years, index=len(years)-1) if years else None
    countries = ["All"] + sorted(data['Country'].dropna().unique().tolist()) if 'Country' in data.columns else ["All"]
    selected_country = st.selectbox("Country", countries, index=0)

# Filter data
df = data.copy()
if selected_year:
    df = df[df['Year'] == selected_year]
if selected_country != "All":
    df = df[df['Country'] == selected_country]

# Model setup
feature_cols = ["Incidents","Fatalities","Injuries","Hostages","Rank"]
X_all = data.reindex(columns=feature_cols).fillna(0)
y_all = data["Score"].fillna(0) if "Score" in data.columns else pd.Series(np.zeros(len(X_all)))
model = RandomForestRegressor(n_estimators=150, random_state=42)
model_trained = False

if role == "admin":
    # Admin can retrain model
    retrain = st.checkbox("Retrain model fully (slow)", value=False)
    if retrain:
        with st.spinner("Training model on full dataset..."):
            try:
                model.fit(X_all, y_all)
                model_trained = True
                st.success("Model trained on full dataset.")
            except Exception as e:
                st.error(f"Model training failed: {e}")
    else:
        # Quick train on sample for responsiveness
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

    # Prediction panel
    st.subheader("🤖 Score Estimator — Quick Prediction")
    left_p, right_p = st.columns([1, 1])
    with left_p:
        i = st.number_input("Incidents", min_value=0, max_value=100000, value=200)
        f = st.number_input("Fatalities", min_value=0, max_value=100000, value=50)
        inj = st.number_input("Injuries", min_value=0, max_value=100000, value=100)
    with right_p:
        h = st.number_input("Hostages", min_value=0, max_value=10000, value=5)
        r = st.slider("Rank", 1, 300, 50)

    if st.button("Predict Score"):
        if not model_trained:
            st.error("Model is not trained. Please enable retrain or provide more data.")
        else:
            pred = model.predict([[i, f, inj, h, r]])[0]
            st.markdown(f"<div style='font-size:20px; color:#E6B85A; font-weight:600'>🌋 Predicted Score: {pred:.2f}</div>", unsafe_allow_html=True)

    # Full dashboard content - maps, trends, etc.
    st.subheader(f"🗺️ Global Terrorism Score — {selected_year if selected_year else 'All Years'}")
    if {'Country', 'iso3c'}.issubset(df.columns):
        map_df = df.dropna(subset=['iso3c']).groupby(['Country', 'iso3c'], as_index=False).agg({"Score":"mean", "Fatalities":"sum", "Incidents":"sum"})
        if not map_df.empty:
            choropleth = px.choropleth(
                map_df,
                locations="iso3c",
                color="Score",
                hover_name="Country",
                hover_data={"Fatalities":True, "Incidents":True, "iso3c":False},
                color_continuous_scale=px.colors.sequential.Plasma,
                projection="natural earth",
            )
            choropleth.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                                     paper_bgcolor="rgba(0,0,0,0)",
                                     plot_bgcolor="rgba(0,0,0,0)",
                                     coloraxis_colorbar=dict(title="Score"))
            st.plotly_chart(choropleth, use_container_width=True)
        else:
            st.info("No data for selected filters or missing iso3c codes.")
    else:
        st.info("Required columns missing for map.")

    if st.checkbox("Show multi-year trend chart", value=True):
        trend_df = data.groupby("Year").agg({"Incidents":"sum", "Fatalities":"sum", "Score":"mean"}).reset_index()
        if trend_df.empty:
            st.info("No year-wise data available.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend_df["Year"], y=trend_df["Incidents"], name="Incidents", mode="lines+markers", line=dict(width=2)))
            fig.add_trace(go.Scatter(x=trend_df["Year"], y=trend_df["Fatalities"], name="Fatalities", mode="lines+markers", line=dict(width=2)))
            fig.update_layout(height=360, margin=dict(t=10, b=0, l=0, r=0), legend=dict(orientation="h"), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

elif role == "user":
    st.title(f"Welcome User {name} — Read-Only Dashboard")
    st.dataframe(df)

    st.subheader("Restricted Analytics View")
    st.markdown("As a user, you cannot retrain the model or make changes.")
    if 'Country' in df.columns and 'Fatalities' in df.columns:
        top10 = df.groupby("Country")["Fatalities"].sum().nlargest(10).reset_index()
        if not top10.empty:
            bar = px.bar(top10.sort_values("Fatalities"), x="Fatalities", y="Country", orientation="h",
                         color="Fatalities", color_continuous_scale=px.colors.sequential.Reds)
            bar.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=360, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(bar, use_container_width=True)
        else:
            st.info("No data available for fatalities chart.")
    else:
        st.info("Required columns missing for user charts.")

# Logout button
authenticator.logout("Logout", "sidebar")

# Footer
st.markdown("<div class='footer'>✨ © 2025 Vaishnavi Raut — AI-Powered Terrorism Analytics (Premium)</div>", unsafe_allow_html=True)
