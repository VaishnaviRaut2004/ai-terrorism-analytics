# step8_ai_dashboard_v2.py
# Emerald Matrix — Dark Premium Dashboard (Final Upgrade)
# PIN Auth with Roles: Admin (edit) / Viewer (read-only)
# Features: Dark theme, glassmorphism, animations, searchable dropdowns, admin edit controls, viewer read-only
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
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Emerald Matrix — Dark Premium", layout="wide", page_icon="🟢")

# -------------------------
# USERS with ROLES
# -------------------------
USERS = {
    "vaishnavi": {"name": "Vaishnavi Raut", "pin": "1981", "role": "admin"},
    "viewer": {"name": "Read Only", "pin": "0000", "role": "viewer"}
}

# -------------------------
# SESSION STATE INIT
# -------------------------
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False
if "username" not in st.session_state:
    st.session_state.username = None
if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0

# -------------------------
# LOGIN / LOGOUT Helpers
# -------------------------
def login_widget():
    st.markdown(
        """
        <div style='display:flex; gap:12px; align-items:center;'>
          <h2 style='margin:0; color:#7fffd4;'>🔐 Sign in</h2>
          <div style='color:#9fd8c9; margin-left:6px; font-size:13px;'>(username & PIN)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([2,1])
    with col1:
        user = st.text_input("Username", key="login_user", placeholder="vaishnavi")
    with col2:
        pin = st.text_input("PIN", key="login_pin", type="password", placeholder="1981")

    if st.button("Login", key="login_btn"):
        st.session_state.login_attempts += 1
        if user in USERS and pin == USERS[user]["pin"]:
            st.session_state.auth_ok = True
            st.session_state.username = user
            st.success(f"Welcome, {USERS[user]['name']} ({USERS[user]['role'].upper()})")
            time.sleep(0.6)  # small pause for UX
        else:
            st.error("Invalid username or PIN.")
            if st.session_state.login_attempts >= 5:
                st.warning("Too many failed attempts. Refresh to try again.")


def logout():
    st.session_state.auth_ok = False
    st.session_state.username = None

# -------------------------
# SHOW LOGIN
# -------------------------
if not st.session_state.auth_ok:
    # full-screen glass login with background
    st.markdown(
        """
        <style>
        .bg {background: radial-gradient(circle at 10% 10%, rgba(22,128,106,0.18), transparent 10%),
                     linear-gradient(180deg,#021213, #04221f); min-height: 420px; border-radius:14px; padding:18px;}
        .login-card {max-width:920px; margin:40px auto; padding:24px; border-radius:14px; backdrop-filter: blur(6px);
                     background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border:1px solid rgba(127,255,212,0.04);}
        .title {color:#7fffd4; margin:0}
        .subtitle {color:#9fd8c9; margin-top:6px}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='login-card'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>🟢 Emerald Matrix — Secure Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Dark premium analytics • Login to continue</div>", unsafe_allow_html=True)
    st.write("")
    login_widget()
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# -------------------------
# Post-login: role determination
# -------------------------
current_user = st.session_state.username
role = USERS.get(current_user, {}).get("role", "viewer")

# -------------------------
# THEME CSS — Dark Emerald Premium
# -------------------------
st.markdown(
    """
    <style>
    :root{
        --bg:#031617;
        --card: rgba(255,255,255,0.03);
        --glass: rgba(255,255,255,0.02);
        --emerald:#1dbf9b;
        --teal:#0fb5a3;
        --muted:#9bd2c7;
    }
    .stApp { background: linear-gradient(180deg,var(--bg), #052828); color: #dffefa; font-family: Inter, ui-sans-serif, system-ui; }
    .topbar { padding:12px 18px; margin-bottom:12px; border-radius:10px; display:flex; justify-content:space-between; align-items:center;
              background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); box-shadow: 0 6px 30px rgba(0,0,0,0.6); border:1px solid rgba(255,255,255,0.02)}
    .card { background: var(--card); border-radius:12px; padding:12px; margin-bottom:12px; border:1px solid rgba(255,255,255,0.02); box-shadow: 0 8px 40px rgba(3,60,37,0.2);}
    .glass { backdrop-filter: blur(6px); }
    .metric { padding:10px; border-radius:8px; }
    .muted { color:var(--muted); font-size:13px; }
    .chip { padding:6px 10px; border-radius:999px; background: rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.02); }
    .btn { background: linear-gradient(90deg, var(--emerald), var(--teal)); color: #001; padding:8px 12px; border-radius:9px; border:none; font-weight:600}
    .small { font-size:12px; color:#9bd2c7 }
    .sidebar-icon { font-size:18px; margin-right:8px }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Header
# -------------------------
st.markdown(
    f"""
    <div class='topbar'>
      <div>
        <h2 style='margin:0; color:var(--emerald)'>🟢 Emerald Matrix — Premium Analytics</h2>
        <div class='muted'>User: <b>{USERS[current_user]['name']}</b> • Role: <b>{role.upper()}</b></div>
      </div>
      <div style='display:flex; gap:10px; align-items:center;'>
        <div class='chip'>Data: gti_cleaned.csv</div>
        <div class='chip'>Theme: Dark Emerald</div>
        <div>
        </div>
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
# Load dataset
# -------------------------
DATA_FILE = "gti_cleaned.csv"
if not Path(DATA_FILE).exists():
    st.error("❌ Dataset 'gti_cleaned.csv' not found. Place it in app root.")
    st.stop()

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

data = load_data(DATA_FILE)

# Ensure iso3
if "iso3c" not in data.columns or data["iso3c"].isnull().any():
    if "Country" in data.columns:
        def _safe_iso(x):
            try:
                if pd.isna(x):
                    return None
                s = str(x).strip()
                if len(s) == 3 and s.isalpha():
                    return s.upper()
                return pycountry.countries.lookup(s).alpha_3
            except:
                return None
        data["iso3c"] = data["Country"].apply(_safe_iso)
    else:
        data["iso3c"] = None

# Numeric fixes
for c in ["Incidents","Fatalities","Injuries","Hostages","Score","Rank","Year"]:
    if c in data.columns:
        data[c] = pd.to_numeric(data[c], errors='coerce')
for c in ["Incidents","Fatalities","Injuries","Hostages","Score","Rank"]:
    if c in data.columns:
        data[c] = data[c].fillna(0)

# -------------------------
# Sidebar (with icons) - role aware
# -------------------------
with st.sidebar:
    st.markdown("<div style='padding:8px; border-radius:8px' class='glass'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:var(--emerald); margin:0'>🔎 Controls</h3>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Filters & visualization settings</div>", unsafe_allow_html=True)
    st.write("")

    # Year dropdown with search helper
    years = sorted(data['Year'].dropna().astype(int).unique().tolist()) if 'Year' in data.columns else []
    year_search = st.text_input('Search year', key='year_search')
    year_options = [y for y in years if year_search.strip() in str(y)] if year_search else years
    selected_year = st.selectbox('Year', year_options, index=len(year_options)-1 if year_options else 0, key='sel_year', disabled=(role=='viewer'))

    # Country dropdown with search
    countries = sorted(data['Country'].dropna().unique().tolist()) if 'Country' in data.columns else []
    country_search = st.text_input('Search country', key='country_search')
    country_options = [c for c in countries if country_search.lower() in c.lower()] if country_search else countries
    selected_country = st.selectbox('Country', ['All'] + country_options, index=0, key='sel_country', disabled=(role=='viewer'))

    st.markdown('---')
    st.subheader('Visuals')
    top_n = st.slider('Top N', 5, 30, 10, key='top_n', disabled=(role=='viewer'))
    show_anim = st.checkbox('Animated map (year)', value=True, key='show_anim', disabled=(role=='viewer'))
    show_bubble = st.checkbox('Bubble chart', value=True, key='show_bubble', disabled=(role=='viewer'))
    show_heat = st.checkbox('Heatmap', value=True, key='show_heat', disabled=(role=='viewer'))
    st.markdown('---')
    st.subheader('Model')
    if role == 'admin':
        retrain = st.checkbox('Retrain model (slow)', value=False, key='retrain')
        n_est = st.number_input('RF n_estimators', min_value=10, max_value=500, value=150, key='n_est')
    else:
        st.markdown("<div class='small'>Viewer: model controls are disabled.</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Apply filters to df
df = data.copy()
if 'Year' in df.columns:
    try:
        df = df[df['Year'] == int(selected_year)]
    except Exception:
        pass
if selected_country and selected_country != 'All':
    df = df[df['Country'] == selected_country]

# -------------------------
# Main metrics
# -------------------------
col1, col2, col3, col4 = st.columns([2,1,1,1])
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    top_country = data.groupby('Country')['Incidents'].sum().idxmax() if 'Country' in data.columns else 'N/A'
    st.markdown(f"<b style='color:var(--emerald)'>{top_country}</b><br><span class='small'>Most affected country (all years)</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    deadliest = int(data.groupby('Year')['Fatalities'].sum().idxmax()) if 'Year' in data.columns else 'N/A'
    st.markdown(f"<b>{deadliest}</b><br><span class='small'>Deadliest year</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    avg_score = data['Score'].mean() if 'Score' in data.columns else 0
    st.markdown(f"<b>{avg_score:.2f}</b><br><span class='small'>Average Score</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with col4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    ccount = data['Country'].nunique() if 'Country' in data.columns else 0
    st.markdown(f"<b>{ccount}</b><br><span class='small'>Countries analyzed</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<br/>', unsafe_allow_html=True)

# -------------------------
# Layout: map / charts
# -------------------------
left, right = st.columns((2,1))
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader('🗺️ Global Terrorism (Score)')
    if show_anim and {'Year','iso3c','Score'}.issubset(data.columns):
        try:
            anim_df = data.groupby(['Year','Country','iso3c'], as_index=False).agg({'Score':'mean','Incidents':'sum'})
            anim_df = anim_df.sort_values('Year')
            fig_map = px.choropleth(anim_df, locations='iso3c', color='Score', hover_name='Country', animation_frame='Year', color_continuous_scale=px.colors.sequential.Plasma, projection='natural earth')
            fig_map.update_layout(margin=dict(t=5,b=0,l=0,r=0))
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception as e:
            st.error('Animated map failed: ' + str(e))
    else:
        map_df = df.groupby(['Country','iso3c'], as_index=False).agg({'Score':'mean','Incidents':'sum'}) if {'Country','iso3c'}.issubset(df.columns) else pd.DataFrame()
        if map_df.empty:
            st.info('No map data for selected filters.')
        else:
            fig = px.choropleth(map_df, locations='iso3c', color='Score', hover_name='Country', color_continuous_scale=px.colors.sequential.Tealgrn)
            fig.update_layout(margin=dict(t=5,b=0,l=0,r=0))
            st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Trend chart
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader('📈 Incidents vs Fatalities (multi-year)')
    if 'Year' in data.columns:
        trend = data.groupby('Year').agg({'Incidents':'sum','Fatalities':'sum'}).reset_index()
        fig_tr = go.Figure()
        fig_tr.add_trace(go.Bar(x=trend['Year'], y=trend['Incidents'], name='Incidents', marker_color='rgba(27, 191, 155, 0.8)'))
        fig_tr.add_trace(go.Line(x=trend['Year'], y=trend['Fatalities'], name='Fatalities', marker_color='rgba(11, 129, 67, 0.9)'))
        fig_tr.update_layout(margin=dict(t=5,b=0,l=0,r=0), legend=dict(orientation='h'))
        st.plotly_chart(fig_tr, use_container_width=True)
    else:
        st.info('Year column not available.')
    st.markdown('</div>', unsafe_allow_html=True)

    # Bubble chart
    if show_bubble:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader('🔵 Fatalities vs Injuries (bubble size = Incidents)')
        if {'Fatalities','Injuries','Incidents','Country'}.issubset(data.columns):
            bubble_df = df.groupby('Country', as_index=False).agg({'Fatalities':'sum','Injuries':'sum','Incidents':'sum'})
            bubble_df = bubble_df[bubble_df['Fatalities'] + bubble_df['Injuries'] > 0]
            if bubble_df.empty:
                st.info('No bubble data for selected filters.')
            else:
                fig_b = px.scatter(bubble_df, x='Injuries', y='Fatalities', size='Incidents', hover_name='Country', size_max=40, color='Incidents', color_continuous_scale=px.colors.sequential.Teal)
                st.plotly_chart(fig_b, use_container_width=True)
        else:
            st.info('Required columns missing for bubble chart.')
        st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader('🏆 Top Countries by Fatalities')
    if 'Country' in df.columns and 'Fatalities' in df.columns:
        top10 = df.groupby('Country')['Fatalities'].sum().nlargest(top_n).reset_index()
        if top10.empty:
            st.info('No data for selected filters.')
        else:
            fig_bar = px.bar(top10.sort_values('Fatalities'), x='Fatalities', y='Country', orientation='h', color='Fatalities', color_continuous_scale=px.colors.sequential.Teal)
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info('No data for this chart.')
    st.markdown('</div>', unsafe_allow_html=True)

    if show_heat:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader('🔥 Heatmap — Year vs Top Countries (Incidents)')
        if 'Year' in data.columns and 'Country' in data.columns:
            top_countries = data.groupby('Country')['Incidents'].sum().nlargest(top_n).index.tolist()
            heat_df = data[data['Country'].isin(top_countries)].pivot_table(values='Incidents', index='Country', columns='Year', aggfunc='sum').fillna(0)
            if heat_df.empty:
                st.info('No data for heatmap.')
            else:
                fig_h = px.imshow(heat_df, labels=dict(x='Year', y='Country', color='Incidents'), text_auto=True, aspect='auto', color_continuous_scale=px.colors.sequential.Mint)
                st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info('Year/Country missing.')
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Prediction panel (role-aware)
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader('🤖 Score Predictor')

# Ensure core cols exist
for c in ['Incidents','Fatalities','Injuries','Hostages','Rank']:
    if c not in data.columns:
        data[c] = 0

X = data[['Incidents','Fatalities','Injuries','Hostages','Rank']].fillna(0)
y = data['Score'].fillna(0) if 'Score' in data.columns else pd.Series(np.zeros(len(X)))

if role == 'admin':
    ca, cb = st.columns(2)
    with ca:
        inc = st.number_input('Incidents', min_value=0, max_value=100000, value=200, key='inc_admin')
        fat = st.number_input('Fatalities', min_value=0, max_value=100000, value=50, key='fat_admin')
        inj = st.number_input('Injuries', min_value=0, max_value=100000, value=100, key='inj_admin')
    with cb:
        host = st.number_input('Hostages', min_value=0, max_value=10000, value=5, key='host_admin')
        rank = st.slider('Rank', 1, 300, 50, key='rank_admin')

    if st.button('Predict (Admin)'):
        model = RandomForestRegressor(n_estimators=(n_est if 'n_est' in locals() else 150), random_state=42)
        sample_n = min(2000, len(X))
        if sample_n > 0:
            Xs = X.sample(sample_n, random_state=42)
            ys = y.loc[Xs.index]
            model.fit(Xs, ys)
            pred = model.predict([[inc, fat, inj, host, rank]])[0]
            st.success(f'Predicted Score: {pred:.2f}')
        else:
            st.warning('Not enough data to train model.')

else:
    st.markdown("<div class='small'>Viewer mode — choose a predefined scenario.</div>", unsafe_allow_html=True)
    scenarios = {
        'Low': [10,0,1,0,100],
        'Medium': [500,20,50,2,60],
        'High': [5000,300,800,20,10]
    }
    sel = st.selectbox('Scenario', list(scenarios.keys()), key='viewer_scn')
    if st.button('Show Prediction (Viewer)'):
        vals = scenarios[sel]
        model = RandomForestRegressor(n_estimators=150, random_state=42)
        sample_n = min(1500, len(X))
        if sample_n > 0:
            Xs = X.sample(sample_n, random_state=42)
            ys = y.loc[Xs.index]
            model.fit(Xs, ys)
            pred = model.predict([vals])[0]
            st.success(f'Predicted Score: {pred:.2f}')
        else:
            st.warning('Not enough data to generate prediction.')

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Admin tools & downloads
# -------------------------
if role == 'admin':
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader('⚙️ Admin Controls')
    csv_full = data.to_csv(index=False).encode()
    st.download_button('Download full dataset', csv_full, file_name='gti_cleaned_full.csv', mime='text/csv')
    st.markdown('</div>', unsafe_allow_html=True)

# Filtered download (all users)
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader('⬇️ Download Filtered Data')
if not df.empty:
    csv_f = df.to_csv(index=False).encode()
    st.download_button('Download filtered CSV', csv_f, file_name=f'gtd_filtered_{selected_year}.csv', mime='text/csv')
else:
    st.info('No data to download for selected filters.')
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Quick insights
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader('🧠 Quick Insights')
ia, ib, ic = st.columns(3)
with ia:
    top_sel = df.groupby('Country')['Incidents'].sum().idxmax() if (not df.empty and 'Country' in df.columns) else 'N/A'
    st.markdown(f'**Top (selected year)**

{top_sel}')
with ib:
    spike = int(data.groupby('Year')['Incidents'].sum().idxmax()) if 'Year' in data.columns else 'N/A'
    st.markdown(f'**Year with spike**

{spike}')
with ic:
    avg_f = data['Fatalities'].mean() if 'Fatalities' in data.columns else 0
    st.markdown(f'**Avg Fatalities (all years)**

{avg_f:.1f}')
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<div style='text-align:center; color:#9bd2c7; padding:10px;'>✨ © 2025 Vaishnavi Raut — Emerald Matrix (Dark Premium)</div>", unsafe_allow_html=True)
