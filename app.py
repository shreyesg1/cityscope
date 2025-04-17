import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import folium_static
import geopandas as gpd
from datetime import datetime, timedelta
import json

from data.data_fetcher import DataFetcher
from models.risk_scorer import RiskScorer
from visualization.visualizer import Visualizer
from utils.helpers import load_config, format_risk_score, generate_summary_statistics

st.set_page_config(
    page_title="CityScope - Urban Risk Intelligence Dashboard",
    layout="wide"
)

# Load configuration
try:
    config = load_config()
except Exception as e:
    config = {
        'map_settings': {
            'center': [40.7128, -74.0060],
            'zoom_start': 11
        }
    }

# Initialize components
try:
    data_fetcher = DataFetcher()
    risk_scorer = RiskScorer()
    visualizer = Visualizer()
except Exception as e:
    st.error("Error initializing components")
    st.stop()

# Title and description
st.title("CityScope - Urban Risk Intelligence Dashboard")
st.markdown("""
    This interactive dashboard helps identify and visualize infrastructure and safety risks across urban neighborhoods.
    Using real-time data from NYC Open Data, we analyze patterns in 311 complaints, crime reports, and health inspections.
""")

# Sidebar controls
with st.sidebar:
    # Time range selector
    days_range = st.slider(
        "Time Range (days)",
        min_value=1,
        max_value=90,
        value=7,
        help="Select the number of days of historical data to analyze"
    )
    
    st.markdown("---")
    
    # Risk weights
    st.markdown("### Risk Weights")
    weights = {
        'crime': st.slider("Crime", 0.0, 1.0, 0.4, format="%.2f"),
        'complaints': st.slider("311", 0.0, 1.0, 0.3, format="%.2f"),
        'health': st.slider("Health", 0.0, 1.0, 0.3, format="%.2f")
    }

# Load data and handle errors
data = None
try:
    data = data_fetcher.get_all_data(days=days_range)
except Exception as e:
    st.error("Error loading data")

# Left side: Risk Map
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Risk Map")
    if data is not None and all(v is not None and not v.empty for v in data.values()):
        try:
            with st.spinner("Generating risk map..."):
                risk_map = visualizer.create_risk_map(
                    data['complaints'],
                    data['crime'],
                    data['health']
                )
                folium_static(risk_map, width=800, height=600)
        except Exception as e:
            st.error("Error creating map")
    else:
        st.info("Waiting for data...")

# Right side: Stats and metrics
with col2:
    st.header("Summary Statistics")
    if data is not None:
        for data_type, df in data.items():
            if df is not None and not df.empty:
                stats = generate_summary_statistics(df)
                
                st.subheader(data_type.title())
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Incidents", f"{stats['total_incidents']:,}")
                with m2:
                    st.metric("Areas", f"{stats['areas_affected']:,}")
                
                if stats['average_risk_score'] is not None:
                    st.metric("Risk Score", f"{stats['average_risk_score']:.1f}")
            else:
                st.info(f"No {data_type} data available")

# Trend charts for each dataset
st.header("Trend Analysis")
if data is not None:
    dataset_configs = {
        '311 Complaints': {
            'key': 'complaints',
            'date_col': 'created_date',
            'category_col': 'complaint_type'
        },
        'Crime Reports': {
            'key': 'crime',
            'date_col': 'cmplnt_fr_dt',
            'category_col': 'ofns_desc'
        },
        'Health Violations': {
            'key': 'health',
            'date_col': 'inspection_date',
            'category_col': 'violation_code'
        }
    }
    
    # Create tab for each dataset type
    tabs = st.tabs(list(dataset_configs.keys()))
    
    for tab, (label, config) in zip(tabs, dataset_configs.items()):
        with tab:
            df = data[config['key']]
            if df is not None and not df.empty:
                with st.spinner(f"Generating {label.lower()} trend chart..."):
                    fig = visualizer.create_trend_chart(
                        df,
                        config['date_col'],
                        config['category_col']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No {label.lower()} data available")

# Hotspot analysis section
st.header("Risk Hotspots")
if data is not None and all(v is not None and not v.empty for v in data.values()):
    try:
        # Combine all incident locations
        valid_coords = []
        dataset_stats = {}
        total_incidents = 0
        
        for data_type, df in data.items():
            if df is not None and not df.empty:
                coords = df[['latitude', 'longitude']].dropna()
                if not coords.empty:
                    valid_coords.append(coords)
                    dataset_stats[data_type] = len(coords)
                    total_incidents += len(coords)

        if valid_coords:
            with st.spinner("Analyzing incident patterns..."):
                all_coords = pd.concat(valid_coords)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Incidents", f"{total_incidents:,}")
                with col2:
                    coverage = (total_incidents - dataset_stats.get('noise', 0)) / total_incidents * 100
                    st.metric("Coverage", f"{coverage:.1f}%")
                
                # Generate and display hotspot map
                hotspots = risk_scorer.detect_hotspots(all_coords)
                
                if hotspots is not None and not hotspots.empty:
                    hotspot_map = visualizer.create_heatmap(hotspots)
                    folium_static(hotspot_map)
                    
                    # Show metrics
                    stats_cols = st.columns(3)
                    with stats_cols[0]:
                        st.metric("Hotspots", len(hotspots))
                    with stats_cols[1]:
                        avg_density = hotspots['density'].mean()
                        st.metric("Avg Density", f"{int(avg_density)}/km²")
                    with stats_cols[2]:
                        st.metric("Avg Area", f"{hotspots['area_km2'].mean():.2f} km²")
                    
                    # Display table
                    hotspot_display = pd.DataFrame({
                        'Incidents': hotspots['count'].map(lambda x: f"{int(x):,}"),
                        'Density (/km²)': hotspots['density'].map(lambda x: f"{int(x):,}"),
                        'Area (km²)': hotspots['area_km2'].map(lambda x: f"{x:.2f}")
                    })
                    st.dataframe(hotspot_display, use_container_width=True)
                else:
                    st.info("No significant risk hotspots detected in the selected time range.")
        else:
            st.warning("Insufficient data for hotspot detection.")
    except Exception as e:
        st.error("Error in hotspot detection")
else:
    st.info("Waiting for data...")

st.markdown("---")
st.markdown(f"Data Source: NYC Open Data | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 