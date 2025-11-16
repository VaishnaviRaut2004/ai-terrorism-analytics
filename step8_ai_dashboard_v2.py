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
    # simply let the UI re-render to show login

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
    .btn-admin { background: linear-gradient(90deg,var(--emerald),var(--teal)); color:white; padding:8px 12px; border-radius:10px; border:none; }
    footer { text-align:center; color:var(--muted); padding:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# HEADER TOPBAR (shows role)
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
        <div>
          <!-- logout button -->
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Logout button (unique key)
if st.button("Logout", key="logout_em"):
    logout()
    st.experimental_rerun()

# -------------------------
# LOAD DATA
# -------------------------
DATA_FILE = "gti_cleaned.csv"
if not Path(DATA_FILE).exists():
    st.error("❌ Dataset 'gti_cleaned.csv' not found in repo root. Please add it.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path)
    return df

data = load_data(DATA_FILE)

# ensure expected columns exist
expected_cols = {"iso3c", "Country", "Rank", "Score", "Incidents", "Fatalities", "Injuries", "Hostages", "Year"}
missing = expected_cols - set(data.columns)
if missing:
    st.warning(f"Missing columns (app will try to continue): {', '.join(sorted(missing))}")

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
for c in ["Incidents", "Fatalities", "Injuries", "Hostages", "Score", "Rank", "Year"]:
    if c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce')

# (fix accidental quote) -> corrected next line
for c in ["Incidents", "Fatalities", "Injuries", "Hostages", "Score", "Rank", "Year"]:
    if c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")
# fill zeros for core numeric columns
for c in ["Incidents", "Fatalities", "Injuries", "Hostages", "Score", "Rank"]:
    if c in data.columns:
        data[c] = data[c].fillna(0)

# -------------------------
# SIDEBAR - role-aware (admins can edit; viewers see disabled controls)
# -------------------------
with st.sidebar:
    st.header("Filters & Settings")
    years = sorted(data["Year"].dropna().astype(int).unique().tolist()) if "Year" in data.columns else []
    # for viewers disable changing year? We'll still let them pick year but it's harmless; main restrictions below
    selected_year = st.selectbox("Year", years, index=len(years)-1 if years else 0, key="sb_year_em", disabled=(role=="viewer"))
    country_list = ["All"] + (sorted(data["Country"].dropna().unique().tolist()) if "Country" in data.columns else [])
    selected_country = st.selectbox("Country", country_list, index=0, key="sb_country_em", disabled=(role=="viewer"))
    st.markdown("---")
    st.subheader("Visualization Options")
    top_n = st.slider("Top N countries", 5, 30, 10, key="sb_topn_em", disabled=(role=="viewer"))
    show_anim_map = st.checkbox("Animated map (year)", value=True, key="sb_anim_em", disabled=(role=="viewer"))
    show_heatmap = st.checkbox("Heatmap (year vs top countries)", value=True, key="sb_heat_em", disabled=(role=="viewer"))
    show_bubble = st.checkbox("Bubble chart (Fatalities vs Injuries)", value=True, key="sb_bub_em", disabled=(role=="viewer"))
    show_pie = st.checkbox("Top-country pie", value=True, key="sb_pie_em", disabled=(role=="viewer"))
    st.markdown("---")
    st.subheader("Model (Admin Only)")
    # only admin can retrain / control model params
    retrain = False
    if role == "admin":
        retrain = st.checkbox("Retrain model (slow)", value=False, key="sb_retrain_em")
        n_estim = st.number_input("RF n_estimators", min_value=10, max_value=1000, value=150, step=10, key="sb_nest_em")
    else:
        st.markdown("<div class='small'>Viewer: model settings are read-only.</div>", unsafe_allow_html=True)

# -------------------------
# Apply filters to df
# -------------------------
df = data.copy()
if "Year" in df.columns:
    df = df[df["Year"] == int(selected_year)]
if selected_country != "All":
    df = df[df["Country"] == selected_country]

# -------------------------
# TOP METRICS (floating cards)
# -------------------------
c1, c2, c3, c4 = st.columns([2,1,1,1])
with c1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    top_country_all = data.groupby("Country")["Incidents"].sum().idxmax() if "Country" in data.columns and not data.empty else "N/A"
    st.markdown(f"<div style='font-size:18px; font-weight:700; color:#0b8043'>{top_country_all}</div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Most affected country (all years)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    deadliest_year = int(data.groupby("Year")["Fatalities"].sum().idxmax()) if "Year" in data.columns and not data.empty else 0
    st.markdown(f"<div style='font-size:16px; font-weight:600'>{deadliest_year}</div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Deadliest year</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    avg_score = data["Score"].mean() if "Score" in data.columns else 0
    st.markdown(f"<div style='font-size:16px; font-weight:600'>{avg_score:.2f}</div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Average Score</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with c4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    country_count = data["Country"].nunique() if "Country" in data.columns else 0
    st.markdown(f"<div style='font-size:16px; font-weight:600'>{country_count}</div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Countries analyzed</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# -------------------------
# MAIN LAYOUT: map and charts
# -------------------------
left, right = st.columns((2,1))
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(f"🗺️ Global Terrorism Score — {selected_year}")
    if show_anim_map and {"iso3c","Year","Score"}.issubset(data.columns):
        try:
            anim_df = data.groupby(["Year","Country","iso3c"], as_index=False).agg({"Score":"mean","Incidents":"sum","Fatalities":"sum"})
            anim_df = anim_df.sort_values("Year")
            map_fig = px.choropleth(anim_df, locations="iso3c", color="Score", hover_name="Country",
                                     animation_frame="Year", color_continuous_scale=px.colors.sequential.Tealgrn,
                                     projection="natural earth")
            map_fig.update_layout(margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(map_fig, use_container_width=True)
        except Exception as e:
            st.error("Animated map failed: " + str(e))
    else:
        map_df = df.groupby(["Country","iso3c"], as_index=False).agg({"Score":"mean","Incidents":"sum","Fatalities":"sum"}) if {"Country","iso3c"}.issubset(df.columns) else pd.DataFrame()
        if map_df.empty:
            st.info("No map data for selected filters.")
        else:
            fig_map_static = px.choropleth(map_df, locations="iso3c", color="Score", hover_name="Country",
                                           color_continuous_scale=px.colors.sequential.Tealgrn, projection="natural earth")
            fig_map_static.update_layout(margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_map_static, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Two-axis trend
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Trend: Incidents vs Fatalities (multi-year)")
    if "Year" in data.columns:
        trend = data.groupby("Year").agg({"Incidents":"sum","Fatalities":"sum","Score":"mean"}).reset_index()
        fig_tr = go.Figure()
        fig_tr.add_trace(go.Bar(x=trend["Year"], y=trend["Incidents"], name="Incidents", marker_color="#66bb6a"))
        fig_tr.add_trace(go.Scatter(x=trend["Year"], y=trend["Fatalities"], name="Fatalities", marker_color="#00796b", yaxis="y2"))
        fig_tr.update_layout(yaxis=dict(title="Incidents"), yaxis2=dict(title="Fatalities", overlaying="y", side="right"),
                             legend=dict(orientation="h"), margin=dict(t=10,b=0))
        st.plotly_chart(fig_tr, use_container_width=True)
    else:
        st.info("Year column not available.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Bubble chart
    if show_bubble:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔵 Fatalities vs Injuries (bubble size = Incidents)")
        if {"Fatalities","Injuries","Incidents","Country"}.issubset(data.columns):
            bubble_df = df.groupby("Country", as_index=False).agg({"Fatalities":"sum","Injuries":"sum","Incidents":"sum"})
            bubble_df = bubble_df[bubble_df["Fatalities"]+bubble_df["Injuries"]>0]
            if bubble_df.empty:
                st.info("No bubble data for selected filters.")
            else:
                bubble = px.scatter(bubble_df, x="Injuries", y="Fatalities", size="Incidents", hover_name="Country", size_max=40, color="Incidents", color_continuous_scale=px.colors.sequential.Teal)
                st.plotly_chart(bubble, use_container_width=True)
        else:
            st.info("Required columns missing for bubble chart.")
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    # Top N bar
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏆 Top Countries by Fatalities")
    if "Country" in df.columns and "Fatalities" in df.columns:
        top10 = df.groupby("Country")["Fatalities"].sum().nlargest(top_n).reset_index()
        if top10.empty:
            st.info("No data for selected filters.")
        else:
            bar = px.bar(top10.sort_values("Fatalities"), x="Fatalities", y="Country", orientation="h", color="Fatalities", color_continuous_scale=px.colors.sequential.Teal)
            st.plotly_chart(bar, use_container_width=True)
    else:
        st.info("No data for this chart.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Heatmap
    if show_heatmap:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔥 Heatmap — Year vs Top Countries (Incidents)")
        if "Year" in data.columns and "Country" in data.columns:
            top_countries = data.groupby("Country")["Incidents"].sum().nlargest(top_n).index.tolist()
            heat_df = data[data["Country"].isin(top_countries)].pivot_table(values="Incidents", index="Country", columns="Year", aggfunc="sum").fillna(0)
            if heat_df.empty:
                st.info("No data for heatmap.")
            else:
                heat = px.imshow(heat_df, labels=dict(x="Year", y="Country", color="Incidents"), text_auto=True, aspect="auto", color_continuous_scale=px.colors.sequential.Mint)
                st.plotly_chart(heat, use_container_width=True)
        else:
            st.info("Year/Country missing.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Correlation
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔬 Correlation (Numeric features)")
    numeric_cols = [c for c in ["Incidents","Fatalities","Injuries","Hostages","Score","Rank"] if c in data.columns]
    if len(numeric_cols) >= 2:
        corr = data[numeric_cols].corr()
        corr_fig = px.imshow(corr, text_auto=True, color_continuous_scale=px.colors.diverging.BrBG)
        st.plotly_chart(corr_fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns.")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# PREDICTION PANEL - ROLE AWARE
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🤖 Score Predictor")

if role == "admin":
    # admin can input custom values
    colA, colB = st.columns(2)
    with colA:
        inc = st.number_input("Incidents", min_value=0, max_value=100000, value=200, key="pred_inc_admin")
        fat = st.number_input("Fatalities", min_value=0, max_value=100000, value=50, key="pred_fat_admin")
        inj = st.number_input("Injuries", min_value=0, max_value=100000, value=100, key="pred_inj_admin")
    with colB:
        host = st.number_input("Hostages", min_value=0, max_value=10000, value=5, key="pred_host_admin")
        rank = st.slider("Rank", 1, 300, 50, key="pred_rank_admin")
    if st.button("Predict (Admin)", key="predict_admin_btn"):
        # prepare model and predict (respect retrain & n_estim)
        feature_cols = ["Incidents","Fatalities","Injuries","Hostages","Rank"]
        for c in feature_cols:
            if c not in data.columns:
                data[c] = 0
        X = data[feature_cols].fillna(0)
        y = data["Score"].fillna(0) if "Score" in data.columns else pd.Series(np.zeros(len(X)))
        model = RandomForestRegressor(n_estimators=int(n_estim) if role=="admin" else 150, random_state=42)
        if retrain:
            with st.spinner("Retraining model on full dataset..."):
                model.fit(X, y)
                st.success("Model retrained.")
        else:
            sample_n = min(2000, len(X))
            if sample_n > 0:
                Xs = X.sample(sample_n, random_state=42)
                ys = y.loc[Xs.index]
                model.fit(Xs, ys)
        pred = model.predict([[inc, fat, inj, host, rank]])[0]
        st.markdown(f"<div style='font-size:20px; color:#0b8043; font-weight:700'>🌋 Predicted Score: {pred:.2f}</div>", unsafe_allow_html=True)

else:
    # viewer: cannot change numbers; provide preset scenarios and readonly predict
    st.markdown("<div class='small'>Viewer mode — choose a predefined scenario to see predicted score.</div>", unsafe_allow_html=True)
    scenarios = {
        "Low (Minor incidents)": {"Incidents":10, "Fatalities":0, "Injuries":1, "Hostages":0, "Rank":100},
        "Medium (Localized spike)": {"Incidents":500, "Fatalities":20, "Injuries":50, "Hostages":2, "Rank":60},
        "High (Major spike)": {"Incidents":5000, "Fatalities":300, "Injuries":800, "Hostages":20, "Rank":10},
        "Custom (Admin only)": {"Incidents":0, "Fatalities":0, "Injuries":0, "Hostages":0, "Rank":50}
    }
    sel = st.selectbox("Scenario", list(scenarios.keys()), key="viewer_scenario")
    btn = st.button("Show Prediction (Viewer)", key="viewer_predict_btn")
    if btn:
        vals = scenarios[sel]
        # prepare model (no retrain for viewer)
        feature_cols = ["Incidents","Fatalities","Injuries","Hostages","Rank"]
        for c in feature_cols:
            if c not in data.columns:
                data[c] = 0
        X = data[feature_cols].fillna(0)
        y = data["Score"].fillna(0) if "Score" in data.columns else pd.Series(np.zeros(len(X)))
        model = RandomForestRegressor(n_estimators=150, random_state=42)
        # quick train on sample
        sample_n = min(1500, len(X))
        if sample_n > 0:
            Xs = X.sample(sample_n, random_state=42)
            ys = y.loc[Xs.index]
            model.fit(Xs, ys)
            pred = model.predict([[vals["Incidents"], vals["Fatalities"], vals["Injuries"], vals["Hostages"], vals["Rank"]]])[0]
            st.markdown(f"<div style='font-size:20px; color:#0b8043; font-weight:700'>🌋 Predicted Score: {pred:.2f}</div>", unsafe_allow_html=True)
        else:
            st.warning("Not enough data to produce prediction.")

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# ADMIN-ONLY PANEL (controls & download)
# -------------------------
if role == "admin":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("⚙️ Admin Controls")
    st.write("You can download the full dataset, retrain model, and change top-N for charts from sidebar.")
    # full data download
    csv_bytes = data.to_csv(index=False).encode()
    st.download_button("Download full dataset (CSV)", csv_bytes, file_name="gti_cleaned_full.csv", mime="text/csv", key="admin_dl_full")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# DOWNLOAD FILTERED CSV (all roles)
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("⬇️ Export Filtered Data")
if not df.empty:
    csvb = df.to_csv(index=False).encode()
    st.download_button("Download filtered CSV", csvb, file_name=f"gtd_filtered_{selected_year}.csv", mime="text/csv", key="dl_filtered_em")
else:
    st.info("No filtered data to download.")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# AUTO INSIGHTS (all roles)
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧠 Quick Insights")
colA, colB, colC = st.columns(3)
with colA:
    top_sel = df.groupby("Country")["Incidents"].sum().idxmax() if (not df.empty and "Country" in df.columns) else "N/A"
    st.markdown(f"**Top (selected year)**\n\n{top_sel}")
with colB:
    spike = int(data.groupby("Year")["Incidents"].sum().idxmax()) if "Year" in data.columns else "N/A"
    st.markdown(f"**Year with spike**\n\n{spike}")
with colC:
    avg_f = data["Fatalities"].mean() if "Fatalities" in data.columns else 0
    st.markdown(f"**Avg Fatalities (all years)**\n\n{avg_f:.1f}")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# FOOTER
# -------------------------
st.markdown("<div style='text-align:center; color:#27664b; padding:10px; font-size:13px;'>✨ © 2025 Vaishnavi Raut — Emerald Matrix Dashboard</div>", unsafe_allow_html=True)
