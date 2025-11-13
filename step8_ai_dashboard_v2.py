# AI-Powered Terrorism Analytics — Modern Dashboard v2.0
# Designed for: Vaishnavi Raut
# Built with Streamlit + Plotly + Scikit-learn

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pycountry
import os

# ------------------------------------------------------------
# 🌈 PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="AI Terrorism Analytics", layout="wide", page_icon="🌍")

# ------------------------------------------------------------
# 🎨 CUSTOM STYLING
# ------------------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 0;
}
.navbar {
    position: sticky;
    top: 0;
    background: rgba(10,10,15,0.85);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding: 12px 25px;
    text-align: center;
    z-index: 9999;
}
.navbar h1 {
    color: #FFD700;
    text-shadow: 0 0 8px rgba(255,215,0,0.6);
    font-size: 30px;
}
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(8px);
    border-radius: 16px;
    padding: 22px;
    margin: 10px 0;
    transition: all 0.3s ease;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
}
.metric {
    background: rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    color: white;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
hr {
    border: 1px solid rgba(255,215,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 🌍 NAVBAR
# ------------------------------------------------------------
st.markdown('<div class="navbar"><h1>🌍 AI-Powered Terrorism Analytics Dashboard</h1></div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# 📂 LOAD DATA
# ------------------------------------------------------------
# ✅ Use relative path so it works on both local and Streamlit Cloud
data_path = "gti_cleaned.csv"

if not os.path.exists(data_path):
    st.error("❌ Dataset not found! Please make sure 'gti_cleaned.csv' is in the same folder as this app.")
    st.stop()

data = pd.read_csv(data_path)

# Add ISO3 country codes safely
def iso3(c):
    try:
        return pycountry.countries.lookup(c).alpha_3
    except:
        return None

if "iso3c" not in data.columns:
    data["iso3c"] = data["Country"].apply(iso3)

# ------------------------------------------------------------
# 🧭 SIDEBAR FILTERS
# ------------------------------------------------------------
st.sidebar.header("🔍 Filters")
year = st.sidebar.selectbox("Select Year", sorted(data["Year"].unique()), index=len(data["Year"].unique()) - 1)
df_year = data[data["Year"] == year]

# ------------------------------------------------------------
# 📊 TABS
# ------------------------------------------------------------
tabs = st.tabs(["🏠 Home", "📈 Visualizations", "🤖 Prediction", "🧠 Insights"])

# ------------------------------------------------------------
# 🏠 HOME TAB
# ------------------------------------------------------------
with tabs[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(f"✨ Overview — Global Terrorism Trends ({year})")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌋 Most Affected Country", data.groupby("Country")["Incidents"].sum().idxmax())
    with col2:
        st.metric("📅 Deadliest Year", int(data.groupby("Year")["Fatalities"].sum().idxmax()))
    with col3:
        st.metric("⚙️ Avg Score", f"{data['Score'].mean():.2f}")
    with col4:
        st.metric("🌐 Countries Analyzed", data['Country'].nunique())

    st.markdown("<br>", unsafe_allow_html=True)

    col_map, col_bar = st.columns((2, 1))
    with col_map:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🗺️ Global Terrorism Score Map")
        map_fig = px.choropleth(
            df_year, locations="iso3c", color="Score",
            hover_name="Country", color_continuous_scale="plasma",
            projection="natural earth"
        )
        st.plotly_chart(map_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_bar:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🏆 Top 10 Countries by Fatalities")
        top10 = df_year.groupby("Country")["Fatalities"].sum().nlargest(10).reset_index()
        bar_fig = px.bar(top10, x="Fatalities", y="Country", orientation="h",
                         color="Fatalities", color_continuous_scale="Reds")
        st.plotly_chart(bar_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# 📈 VISUALIZATION TAB
# ------------------------------------------------------------
with tabs[1]:
    st.subheader("📊 3D & Comparative Visualizations")

    # 3D Scatter
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("### 🌐 3D Scatter — Incidents vs Fatalities vs Score")
    fig3d = px.scatter_3d(
        data, x='Incidents', y='Fatalities', z='Score', color='Country',
        size='Injuries', animation_frame='Year', size_max=22, opacity=0.8,
        color_continuous_scale="viridis"
    )
    fig3d.update_layout(scene=dict(xaxis_title='Incidents', yaxis_title='Fatalities', zaxis_title='Score'))
    st.plotly_chart(fig3d, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 3D Surface
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("### 🔥 3D Surface — Score vs Year vs Rank")
    surf_df = data.pivot_table(values='Score', index='Year', columns='Rank', aggfunc='mean').fillna(0)
    surf_fig = go.Figure(data=[go.Surface(z=surf_df.values, x=surf_df.columns, y=surf_df.index, colorscale='Turbo')])
    surf_fig.update_layout(scene=dict(xaxis_title='Rank', yaxis_title='Year', zaxis_title='Score'))
    st.plotly_chart(surf_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# 🤖 PREDICTION TAB
# ------------------------------------------------------------
with tabs[2]:
    st.subheader("🎯 AI Prediction Model — Terrorism Score Estimator")

    X = data[['Incidents', 'Fatalities', 'Injuries', 'Hostages', 'Rank']].fillna(0)
    y = data['Score'].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("#### Enter Parameters to Predict")
    c1, c2, c3 = st.columns(3)
    with c1:
        i = st.number_input("Incidents", 0, 100000, 200)
    with c2:
        f = st.number_input("Fatalities", 0, 100000, 50)
    with c3:
        inj = st.number_input("Injuries", 0, 100000, 100)
    c4, c5 = st.columns(2)
    with c4:
        h = st.number_input("Hostages", 0, 10000, 5)
    with c5:
        r = st.slider("Rank", 1, 150, 50)

    if st.button("⚙️ Predict Score"):
        pred = model.predict([[i, f, inj, h, r]])[0]
        st.success(f"🌋 Predicted Terrorism Score: **{pred:.2f}**")
        st.balloons()
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# 🧠 INSIGHTS TAB
# ------------------------------------------------------------
with tabs[3]:
    st.subheader("🧠 Key Insights & Correlations")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    hi_country = data.groupby("Country")["Incidents"].sum().idxmax()
    hi_year = int(data.groupby("Year")["Fatalities"].sum().idxmax())
    avg_score = round(data["Score"].mean(), 2)
    total_countries = data["Country"].nunique()

    st.markdown(f"""
    <div style="text-align:center; color:white;">
    <h3 style='color:#FFD700;'>Quick Insights</h3>
    🌋 **Most Affected Country:** {hi_country}<br>
    📅 **Deadliest Year:** {hi_year}<br>
    ⚙️ **Average Score:** {avg_score}<br>
    🌐 **Countries Analyzed:** {total_countries}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.write("### 🔍 Correlation Heatmap")
    corr = data[["Incidents", "Fatalities", "Injuries", "Hostages", "Score", "Rank"]].corr()
    corr_fig = px.imshow(corr, text_auto=True, color_continuous_scale="Tealrose", title="")
    corr_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    st.plotly_chart(corr_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# 🏁 FOOTER
# ------------------------------------------------------------
st.markdown("""
<hr>
<div style='text-align:center; color:#FFD700;'>
✨ © 2025 Vaishnavi Raut | Modern AI Analytics Dashboard ✨
</div>
""", unsafe_allow_html=True)
