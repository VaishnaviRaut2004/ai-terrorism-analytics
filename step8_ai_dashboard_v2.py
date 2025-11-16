# step8_ai_dashboard_v2.py
# AI-Powered Terrorism Analytics — Light Pink-Lavender Aesthetic (Option B)
# PIN Auth + Floating UI + Extra Visualizations + Pop-up Modals
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
import io

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="AI Terrorism Analytics — Aesthetic", layout="wide", page_icon="🌸")

# -------------------------
# PIN AUTH (Option B) - stable
# -------------------------
USERS = {
    "vaishnavi": {"name": "Vaishnavi Raut", "pin": "1981"},
    "demo": {"name": "Demo User", "pin": "0000"}
}

# Session-state init
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False
if "username" not in st.session_state:
    st.session_state.username = None
if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0

def login_widget():
    st.markdown(
        """
        <div style="display:flex; gap:12px; align-items:center;">
          <h2 style="margin:0; color:#7e57c2;">🔐 Sign in</h2>
          <div style="color:#7b6f8f; margin-left:6px; font-size:13px;">(username & 4-digit PIN)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    user_input = st.text_input("Username", key="login_user", placeholder="vaishnavi")
    pin_input = st.text_input("PIN", key="login_pin", type="password", placeholder="1981")
    if st.button("Login", key="login_btn"):
        st.session_state.login_attempts += 1
        if user_input in USERS and pin_input == USERS[user_input]["pin"]:
            st.session_state.auth_ok = True
            st.session_state.username = user_input
            st.success(f"Welcome, {USERS[user_input]['name']}!")
        else:
            st.error("Invalid username or PIN.")
            if st.session_state.login_attempts >= 5:
                st.warning("Too many failed attempts. Refresh the page to try again.")

def logout():
    st.session_state.auth_ok = False
    st.session_state.username = None
    st.experimental_rerun()

# If not authenticated show login
if not st.session_state.auth_ok:
    st.markdown(
        """
        <style>
        .login-panel {
            max-width:900px; margin:28px auto; padding:18px; border-radius:14px;
            background: linear-gradient(180deg, rgba(255,255,255,0.85), rgba(250,245,255,0.85));
            box-shadow: 0 10px 30px rgba(125,95,170,0.08); border:1px solid rgba(126,87,194,0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='login-panel'>", unsafe_allow_html=True)
    st.markdown("<h1 style='color:#7e57c2; margin:0'>🌸 AI-Powered Terrorism Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#7b6f8f; margin-top:6px;'>Aesthetic Light Theme — enter credentials to continue.</p>", unsafe_allow_html=True)
    login_widget()
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# -------------------------
# AUTHENTICATED: Dashboard UI + CSS (Light Pink-Lavender)
# -------------------------
st.markdown(
    """
    <style>
    :root{
        --bg1: #fff7fc;
        --bg2: #f3f0fb;
        --card: rgba(255,255,255,0.9);
        --muted: #7b6f8f;
        --accent1: #f48fb1;
        --accent2: #7e57c2;
        --glass: rgba(255,255,255,0.8);
    }
    .stApp {
        background: linear-gradient(180deg,var(--bg1),var(--bg2));
        color: #2b2b2b;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    .topbar {
        padding:12px 18px; border-radius:12px; margin-bottom:16px;
        background: linear-gradient(90deg, rgba(255,255,255,0.7), rgba(255,240,250,0.7));
        display:flex; justify-content:space-between; align-items:center; box-shadow: 0 8px 30px rgba(125,95,170,0.06);
    }
    .card {
        background: var(--card); border-radius:12px; padding:14px; margin-bottom:12px;
        box-shadow: 0 10px 30px rgba(125,95,170,0.04);
        border: 1px solid rgba(126,87,194,0.06);
        transition: transform .18s ease, box-shadow .2s ease;
    }
    .card:hover { transform: translateY(-6px); box-shadow: 0 18px 40px rgba(125,95,170,0.06);}
    .metric { padding:10px; border-radius:10px; }
    .small { color: var(--muted); font-size:12px; }
    .accent { color: var(--accent2); font-weight:700; }
    .btn { background: linear-gradient(90deg,#f48fb1,#7e57c2); color:white; padding:8px 12px; border-radius:10px; border:none; }
    footer { color: #7b6f8f; text-align:center; padding:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Topbar
st.markdown(
    f"""
    <div class="topbar">
      <div>
        <h2 style="margin:0; color:#7e57c2">🌸 AI-Powered Terrorism Analytics</h2>
        <div style="color:#7b6f8f; font-size:13px;">Light • Aesthetic • Interactive</div>
      </div>
      <div style="display:flex; gap:10px; align-items:center;">
        <div style="background:linear-gradient(90deg,#fff,#fff); padding:6px 10px; border-radius:999px; border:1px solid rgba(126,87,194,0.06); color:var(--muted);">Data: gti_cleaned.csv</div>
        <div style="background:linear-gradient(90deg,#fff,#fff); padding:6px 10px; border-radius:999px; border:1px solid rgba(126,87,194,0.06); color:var(--muted);">User: {USERS.get(st.session_state.username, {}).get('name','User')}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Logout
if st.button("Logout", key="logout_light_theme"):
    logout()

# -------------------------
# Load data
# -------------------------
DATA_FILE = "gti_cleaned.csv"
if not Path(DATA_FILE).exists():
    st.error("❌ Dataset not found. Put 'gti_cleaned.csv' in the app folder.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path)
    return df

data = load_data(DATA_FILE)

# Ensure columns exist
expected = {"iso3c","Country","Rank","Score","Incidents","Fatalities","Injuries","Hostages","Year"}
missing = expected - set(data.columns)
if missing:
    st.warning(f"Expected columns missing: {', '.join(missing)} — app will try to proceed with available columns.")

# iso3 safe fill
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

# numeric conversions
for c in ["Incidents","Fatalities","Injuries","Hostages","Score","Rank","Year"]:
    if c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0)

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("Filters & Options")
    years = sorted(data["Year"].unique().astype(int)) if "Year" in data.columns else []
    selected_year = st.selectbox("Year", years, index=len(years)-1 if years else 0, key="st_year")
    country_list = ["All"] + (sorted(data["Country"].unique().tolist()) if "Country" in data.columns else [])
    selected_country = st.selectbox("Country", country_list, index=0, key="st_country")
    top_n = st.slider("Top N countries (charts)", 5, 30, 10, key="st_topn")
    st.markdown("---")
    st.subheader("Visualization toggles")
    show_anim_map = st.checkbox("Animated world map (year)", value=True, key="st_anim_map")
    show_heatmap = st.checkbox("Show heatmap (year vs country top N)", value=True, key="st_heatmap")
    show_bubble = st.checkbox("Show bubble chart", value=True, key="st_bubble")
    show_pie = st.checkbox("Show top-country pie", value=True, key="st_pie")
    st.markdown("---")
    st.subheader("Model")
    retrain = st.checkbox("Retrain prediction model (slow)", value=False, key="st_retrain")
    st.markdown("<div class='small'>Tip: retrain only if needed.</div>", unsafe_allow_html=True)

# Apply filters
df = data.copy()
if "Year" in df.columns:
    df_year = df[df["Year"] == int(selected_year)]
else:
    df_year = df.copy()
if selected_country != "All":
    df_year = df_year[df_year["Country"] == selected_country]

# -------------------------
# Top metrics (floating cards)
# -------------------------
m1, m2, m3, m4 = st.columns([1.6,1,1,1])
with m1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    top_country_all = data.groupby("Country")["Incidents"].sum().idxmax() if "Country" in data.columns and not data.empty else "N/A"
    st.markdown(f"<div style='font-size:18px; font-weight:700; color:#7e57c2'>{top_country_all}</div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Most affected country (all years)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with m2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    deadliest_year_all = int(data.groupby("Year")["Fatalities"].sum().idxmax()) if "Year" in data.columns and not data.empty else 0
    st.markdown(f"<div style='font-size:16px; font-weight:600'>{deadliest_year_all}</div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Deadliest year</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with m3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    avg_score_all = data["Score"].mean() if "Score" in data.columns else 0
    st.markdown(f"<div style='font-size:16px; font-weight:600'>{avg_score_all:.2f}</div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Average Score</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with m4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    countries_count = data["Country"].nunique() if "Country" in data.columns else 0
    st.markdown(f"<div style='font-size:16px; font-weight:600'>{countries_count}</div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Countries analyzed</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# -------------------------
# Layout: main visualizations
# -------------------------
left, right = st.columns((2,1))

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(f"🗺️ Animated World Map (by Year)" if show_anim_map else f"🗺️ Global Terrorism Score — {selected_year}")

    if show_anim_map and {"iso3c","Year","Score"}.issubset(data.columns):
        try:
            anim_df = data.groupby(["Year","Country","iso3c"], as_index=False).agg({"Score":"mean","Incidents":"sum","Fatalities":"sum"})
            anim_df = anim_df.sort_values("Year")
            map_fig = px.choropleth(anim_df, locations="iso3c", color="Score", hover_name="Country",
                                    animation_frame="Year", color_continuous_scale=px.colors.sequential.Plasma,
                                    projection="natural earth")
            map_fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), coloraxis_colorbar=dict(title="Score"), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(map_fig, use_container_width=True)
        except Exception as e:
            st.error("Animated map failed to render: " + str(e))
    else:
        map_df = df_year.groupby(["Country","iso3c"], as_index=False).agg({"Score":"mean","Incidents":"sum","Fatalities":"sum"}) if {"Country","iso3c"}.issubset(df_year.columns) else pd.DataFrame()
        if map_df.empty:
            st.info("No data for map (check filters).")
        else:
            fig_map_static = px.choropleth(map_df, locations="iso3c", color="Score", hover_name="Country",
                                           color_continuous_scale=px.colors.sequential.Plasma, projection="natural earth")
            fig_map_static.update_layout(margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_map_static, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Two-axis trend chart
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Trend (Incidents vs Fatalities) — Multi-year")
    if "Year" in data.columns:
        trend = data.groupby("Year").agg({"Incidents":"sum","Fatalities":"sum","Score":"mean"}).reset_index()
        fig_tr = go.Figure()
        fig_tr.add_trace(go.Bar(x=trend["Year"], y=trend["Incidents"], name="Incidents", marker_color="#f48fb1", yaxis="y1"))
        fig_tr.add_trace(go.Line(x=trend["Year"], y=trend["Fatalities"], name="Fatalities", marker_color="#7e57c2", yaxis="y2"))
        fig_tr.update_layout(yaxis=dict(title="Incidents"), yaxis2=dict(title="Fatalities", overlaying="y", side="right"),
                             legend=dict(orientation="h"), margin=dict(t=10,b=0))
        st.plotly_chart(fig_tr, use_container_width=True)
    else:
        st.info("Year column not available.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Bubble chart: Fatalities vs Injuries
    if show_bubble:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔵 Fatalities vs Injuries (Bubble size = Incidents)")
        if {"Fatalities","Injuries","Incidents","Country"}.issubset(data.columns):
            bubble_df = df_year.groupby("Country", as_index=False).agg({"Fatalities":"sum","Injuries":"sum","Incidents":"sum"})
            bubble_df = bubble_df[bubble_df["Fatalities"]+bubble_df["Injuries"]>0]
            if bubble_df.empty:
                st.info("No data for bubble chart for selected filters.")
            else:
                bubble = px.scatter(bubble_df, x="Injuries", y="Fatalities", size="Incidents", hover_name="Country", size_max=40,
                                    color="Fatalities", color_continuous_scale=px.colors.sequential.Pinkyl)
                st.plotly_chart(bubble, use_container_width=True)
        else:
            st.info("Required columns missing for bubble chart.")
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    # Top-N pie
    if show_pie:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🥧 Top Countries by Incidents (Pie)")
        if "Country" in df_year.columns and "Incidents" in df_year.columns:
            topn = df_year.groupby("Country")["Incidents"].sum().nlargest(top_n).reset_index()
            if topn.empty:
                st.info("No data for pie chart.")
            else:
                pie = px.pie(topn, names="Country", values="Incidents", hole=0.45)
                pie.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(pie, use_container_width=True)
        else:
            st.info("No data for pie chart.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Heatmap year vs country (top N)
    if show_heatmap:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔥 Heatmap — Year vs Top Countries (Incidents)")
        if "Year" in data.columns and "Country" in data.columns:
            top_countries = data.groupby("Country")["Incidents"].sum().nlargest(top_n).index.tolist()
            heat_df = data[data["Country"].isin(top_countries)].pivot_table(values="Incidents", index="Country", columns="Year", aggfunc="sum").fillna(0)
            if heat_df.empty:
                st.info("No data for heatmap.")
            else:
                heat = px.imshow(heat_df, labels=dict(x="Year", y="Country", color="Incidents"), text_auto=True, aspect="auto",
                                 color_continuous_scale=px.colors.sequential.thermal)
                heat.update_layout(height=360, margin=dict(t=10,b=0))
                st.plotly_chart(heat, use_container_width=True)
        else:
            st.info("Year/Country columns missing.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Correlation heatmap
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔬 Correlation — Numeric Features")
    numeric_cols = [c for c in ["Incidents","Fatalities","Injuries","Hostages","Score","Rank"] if c in data.columns]
    if len(numeric_cols) >= 2:
        corr = data[numeric_cols].corr()
        corr_fig = px.imshow(corr, text_auto=True, color_continuous_scale=px.colors.diverging.RdBu)
        corr_fig.update_layout(height=300, margin=dict(t=10,b=0))
        st.plotly_chart(corr_fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns.")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Modal drilldown: show details for a selected country
# -------------------------
st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
if "Country" in data.columns:
    with st.expander("🔎 Open country drill-down (modal)"):
        sel_country = st.selectbox("Choose country for details", ["All"] + sorted(data["Country"].unique().tolist()), key="modal_country_sel")
        if st.button("Open Country Details", key="open_country_modal"):
            # Use Streamlit modal
            with st.modal("Country Details", key="country_modal"):
                st.header(f"Details — {sel_country}")
                if sel_country == "All":
                    st.write("Showing summary for all countries.")
                    st.dataframe(data.head(200))
                else:
                    cdf = data[data["Country"] == sel_country]
                    st.markdown(f"**Total incidents:** {int(cdf['Incidents'].sum())}")
                    st.markdown(f"**Total fatalities:** {int(cdf['Fatalities'].sum())}")
                    st.markdown(f"**Avg score:** {cdf['Score'].mean():.2f}")
                    st.plotly_chart(px.line(cdf.groupby("Year").agg({"Incidents":"sum","Fatalities":"sum"}).reset_index(), x="Year", y="Incidents", title="Incidents over years"), use_container_width=True)
                    st.download_button("Download country data (CSV)", cdf.to_csv(index=False), file_name=f"{sel_country}_data.csv", mime="text/csv")

# -------------------------
# Prediction panel (floating)
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🤖 Quick Score Predictor")
colp1, colp2 = st.columns([1,1])
with colp1:
    pred_inc = st.number_input("Incidents", 0, 100000, 200, key="pred_inc")
    pred_fat = st.number_input("Fatalities", 0, 100000, 50, key="pred_fat")
    pred_inj = st.number_input("Injuries", 0, 100000, 100, key="pred_inj")
with colp2:
    pred_host = st.number_input("Hostages", 0, 10000, 5, key="pred_host")
    pred_rank = st.slider("Rank", 1, 300, 50, key="pred_rank")

feature_cols = [c for c in ["Incidents","Fatalities","Injuries","Hostages","Rank"] if c in data.columns]
if len(feature_cols) < 5:
    # still allow prediction if some columns missing by filling zeros
    for c in ["Incidents","Fatalities","Injuries","Hostages","Rank"]:
        if c not in data.columns:
            data[c] = 0

X = data[["Incidents","Fatalities","Injuries","Hostages","Rank"]].fillna(0)
y = data["Score"].fillna(0) if "Score" in data.columns else pd.Series(np.zeros(len(X)))

model = RandomForestRegressor(n_estimators=150, random_state=42)
if retrain:
    with st.spinner("Training model on full dataset..."):
        model.fit(X, y)
        st.success("Model trained on full dataset.")
else:
    sample_n = min(2000, len(X))
    if sample_n > 0:
        with st.spinner("Training quick sample model..."):
            Xs = X.sample(sample_n, random_state=42)
            ys = y.loc[Xs.index]
            model.fit(Xs, ys)
    else:
        st.warning("No data available for training predictor.")

if st.button("Predict", key="predict_btn"):
    try:
        pred_value = model.predict([[pred_inc, pred_fat, pred_inj, pred_host, pred_rank]])[0]
        st.markdown(f"<div style='font-size:20px; color:#7e57c2; font-weight:700'>🌋 Predicted Score: {pred_value:.2f}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error("Prediction failed: " + str(e))
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Download filtered data
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("⬇️ Export Data")
if not df_year.empty:
    csv_bytes = df_year.to_csv(index=False).encode()
    st.download_button("Download filtered CSV", data=csv_bytes, file_name=f"gtd_filtered_{selected_year}.csv", mime="text/csv", key="download_filtered")
else:
    st.info("No data to download for selected filters.")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Footer
# -------------------------
st.markdown("<div style='text-align:center; color:#7b6f8f; padding:10px; font-size:13px;'>✨ © 2025 Vaishnavi Raut — Aesthetic Dashboard</div>", unsafe_allow_html=True)
