# ============================================
#   RIC DASHBOARD – Terrorism Analytics
#   Rich • Interactive • Clean UI
# ============================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------
# RIC CSS Styling
# ---------------------------------------------------------------
def apply_ric_theme_css():
    st.markdown("""
    <style>
    /* Main container adjustments */
    .block-container {
        padding-top: 1.3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #1A1A1F;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.35);
    }

    /* DataFrame header */
    .dataframe th {
        background-color: #7B5CFF !important;
        color: white !important;
        font-weight: bold;
    }

    /* Card styling */
    .stCard {
        background-color: #1A1A1F;
        padding: 20px;
        border-radius: 18px;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.40);
        margin-bottom: 18px;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
st.sidebar.title("🔍 RIC Dashboard Filters")
uploaded = st.sidebar.file_uploader("Upload Terrorism Dataset (CSV)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.info("RIC Dashboard • Premium UI")


# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------
if uploaded is not None:
    df = pd.read_csv(uploaded)

    st.title("🔥 RIC Terrorism Analytics Dashboard")
    apply_ric_theme_css()

    st.markdown("### **📁 Dataset Overview**")
    st.dataframe(df.head())

    # ------------------------------------------
    # Filters
    # ------------------------------------------
    st.sidebar.subheader("Filter Data")

    years = st.sidebar.multiselect("Select Year", sorted(df["Year"].unique()))
    countries = st.sidebar.multiselect("Select Country", sorted(df["Country"].unique()))
    attack_types = st.sidebar.multiselect("Select Attack Type", sorted(df["AttackType"].unique()))

    data = df.copy()

    if years:
        data = data[data["Year"].isin(years)]
    if countries:
        data = data[data["Country"].isin(countries)]
    if attack_types:
        data = data[data["AttackType"].isin(attack_types)]

    # ------------------------------------------
    # Metrics Row
    # ------------------------------------------
    st.markdown("### 📊 **Key Metrics**")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Attacks", len(data))

    with col2:
        st.metric("Total Killed", int(data["Fatalities"].sum()))

    with col3:
        st.metric("Total Injured", int(data["Injuries"].sum()))

    with col4:
        st.metric("Countries Affected", data["Country"].nunique())

    st.markdown("---")

    # ------------------------------------------
    # Attack Trends Over Time
    # ------------------------------------------
    st.markdown("### 📈 **Attack Trends Over Time**")
    yearly = data.groupby("Year").size().reset_index(name="Attacks")

    fig1 = px.line(yearly, x="Year", y="Attacks", markers=True,
                   title="Yearly Attack Trend")
    st.plotly_chart(fig1, use_container_width=True)

    # ------------------------------------------
    # Attacks by Country
    # ------------------------------------------
    st.markdown("### 🌍 **Most Affected Countries**")
    top_countries = data["Country"].value_counts().head(10).reset_index()
    top_countries.columns = ["Country", "Attacks"]

    fig2 = px.bar(top_countries, x="Country", y="Attacks",
                  title="Top 10 Countries by Attack Count")
    st.plotly_chart(fig2, use_container_width=True)

    # ------------------------------------------
    # Attack Type Distribution
    # ------------------------------------------
    st.markdown("### 🎯 **Attack Type Distribution**")
    attack_type_dist = data["AttackType"].value_counts().reset_index()
    attack_type_dist.columns = ["AttackType", "Count"]

    fig3 = px.pie(attack_type_dist, names="AttackType", values="Count",
                  title="Distribution of Attack Types")
    st.plotly_chart(fig3, use_container_width=True)

    # ------------------------------------------
    # Target Type Analysis
    # ------------------------------------------
    st.markdown("### 🎯 **Target Analysis**")
    if "TargetType" in data.columns:
        target_count = data["TargetType"].value_counts().head(8).reset_index()
        target_count.columns = ["TargetType", "Count"]

        fig4 = px.bar(target_count, x="TargetType", y="Count",
                      title="Top Target Types")
        st.plotly_chart(fig4, use_container_width=True)

    # ------------------------------------------
    # Region Map
    # ------------------------------------------
    st.markdown("### 🗺️ **Global Attack Heatmap**")

    if "Latitude" in data.columns and "Longitude" in data.columns:
        fig5 = px.scatter_mapbox(
            data,
            lat="Latitude",
            lon="Longitude",
            color="AttackType",
            zoom=1,
            height=500,
            hover_name="Country",
        )
        fig5.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------
    # Detailed Table
    # ------------------------------------------
    st.markdown("### 📄 Detailed Records")
    st.dataframe(data)

else:
    st.title("🔥 RIC Terrorism Analytics Dashboard")
    st.info("Upload a dataset from the sidebar to begin.")
