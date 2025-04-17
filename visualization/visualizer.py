import folium
from folium import plugins
from folium.plugins import HeatMap, MarkerCluster, HeatMapWithTime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from streamlit_folium import folium_static
from branca.colormap import LinearColormap
import streamlit as st
from datetime import datetime, timedelta

class Visualizer:
    def __init__(self):
        """
        Initialize the Visualizer with map settings and styling configurations.

        Args:
            None

        Returns:
            None
        """
        self.map_center = [40.7128, -74.0060]  # NYC coordinates
        self.zoom_start = 11
        
        # Define borough coordinates for context
        self.borough_coords = {
            'MANHATTAN': [40.7831, -73.9712],
            'BROOKLYN': [40.6782, -73.9442],
            'QUEENS': [40.7282, -73.7949],
            'BRONX': [40.8448, -73.8648],
            'STATEN ISLAND': [40.5795, -74.1502]
        }
        
        self.color_schemes = {
            'complaints': LinearColormap(
                colors=['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594'],
                vmin=0, vmax=1,
                caption='311 Complaints Density'
            ),
            'crime': LinearColormap(
                colors=['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#005a32'],
                vmin=0, vmax=1,
                caption='Crime Reports Density'
            ),
            'health': LinearColormap(
                colors=['#fff5eb', '#fee6ce', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#8c2d04'],
                vmin=0, vmax=1,
                caption='Health Violations Density'
            )
        }
        
        self.cluster_icon_functions = {
            'complaints': """
                function(cluster) {
                    var count = cluster.getChildCount();
                    var size = count < 100 ? 30 : count < 1000 ? 35 : 40;
                    return L.divIcon({
                        html: '<div style="background-color: rgba(66, 146, 198, 0.9); color: white; border-radius: 50%; width: ' + size + 'px; height: ' + size + 'px; text-align: center; line-height: ' + size + 'px; font-weight: bold; box-shadow: 0 0 10px rgba(0,0,0,0.3);">' + count + '</div>',
                        className: 'marker-cluster-complaints',
                        iconSize: L.point(size, size)
                    });
                }
            """,
            'crime': """
                function(cluster) {
                    var count = cluster.getChildCount();
                    var size = count < 100 ? 30 : count < 1000 ? 35 : 40;
                    return L.divIcon({
                        html: '<div style="background-color: rgba(35, 139, 69, 0.9); color: white; border-radius: 50%; width: ' + size + 'px; height: ' + size + 'px; text-align: center; line-height: ' + size + 'px; font-weight: bold; box-shadow: 0 0 10px rgba(0,0,0,0.3);">' + count + '</div>',
                        className: 'marker-cluster-crime',
                        iconSize: L.point(size, size)
                    });
                }
            """,
            'health': """
                function(cluster) {
                    var count = cluster.getChildCount();
                    var size = count < 100 ? 30 : count < 1000 ? 35 : 40;
                    return L.divIcon({
                        html: '<div style="background-color: rgba(217, 72, 1, 0.9); color: white; border-radius: 50%; width: ' + size + 'px; height: ' + size + 'px; text-align: center; line-height: ' + size + 'px; font-weight: bold; box-shadow: 0 0 10px rgba(0,0,0,0.3);">' + count + '</div>',
                        className: 'marker-cluster-health',
                        iconSize: L.point(size, size)
                    });
                }
            """
        }
    
    def create_base_map(self, center=None, zoom_start=None):
        """
        Create a base Folium map centered on NYC
        """
        center = center or self.map_center
        zoom_start = zoom_start or self.zoom_start
        
        return folium.Map(
            location=center,
            zoom_start=zoom_start,
            tiles='cartodbpositron'
        )
    
    def create_risk_map(self, complaints_df, crime_df, health_df):
        """
        Create a folium map with multiple layers of risk data.

        Args:
            complaints_df: DataFrame containing 311 complaints data
            crime_df: DataFrame containing crime report data
            health_df: DataFrame containing health violation data

        Returns:
            Folium map object with risk layers
        """
        try:
            # Create base map with a modern style
            m = self.create_base_map()
            
            # Add borough labels
            for borough, coords in self.borough_coords.items():
                folium.Marker(
                    coords,
                    icon=folium.DivIcon(
                        html=f'<div style="font-size: 14px; font-weight: bold; color: #666;">{borough}</div>'
                    )
                ).add_to(m)
            
            # Function to create heatmap layer
            @st.cache_data(ttl=3600, show_spinner=False)
            def create_heatmap_data(df, name):
                if df is not None and not df.empty:
                    # Sample data for better performance if dataset is large
                    if len(df) > 10000:
                        df = df.sample(n=10000, random_state=42)
                    
                    data = df[['latitude', 'longitude']].dropna()
                    if not data.empty:
                        data = data.astype(float).values.tolist()
                        layer = folium.FeatureGroup(name=name)
                        HeatMap(
                            # Reduce if performance is poor
                            data,
                            min_opacity=0.4,
                            radius=20,
                            blur=15,
                            max_zoom=13
                        ).add_to(layer)
                        return layer
                return None
            
            # Add layers using cached function
            layers = {
                '311 Complaints Heat': complaints_df,
                'Crime Reports Heat': crime_df,
                'Health Violations Heat': health_df
            }
            
            for name, df in layers.items():
                layer = create_heatmap_data(df, name)
                if layer:
                    layer.add_to(m)
            
            # Add layer control
            folium.LayerControl(
                position='topright',
                collapsed=False
            ).add_to(m)
            
            return m
            
        except Exception as e:
            st.error(f"Error in create_risk_map: {str(e)}")
            raise e
    
    def create_trend_chart(self, df, date_column, category_column):
        """
        Create a line chart showing trends over time.

        Args:
            df: DataFrame containing time series data
            date_column: Name of the date column
            category_column: Name of the category column

        Returns:
            Plotly figure object
        """
        if df is None or df.empty:
            return None
            
        # Ensure we're not showing future dates
        current_time = pd.Timestamp.now()
        df = df[df[date_column] <= current_time].copy()
        
        # Group by date and category
        daily_counts = df.groupby([
            pd.Grouper(key=date_column, freq='6H'),
            category_column
        ]).size().reset_index(name='count')
        
        # Get top 10 categories by total count
        top_categories = df[category_column].value_counts().nlargest(10).index
        daily_counts = daily_counts[daily_counts[category_column].isin(top_categories)]
        
        # Create chart
        fig = px.line(
            daily_counts,
            x=date_column,
            y='count',
            color=category_column,
            title=f'Incident Trends by Type (Top 10 Categories)',
            labels={
                date_column: 'Date',
                'count': 'Number of Incidents',
                category_column: 'Incident Type'
            }
        )
        
        # Setup plot
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_x=0.5,
            title_font_size=20,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=1.1,
                xanchor="left",
                x=0.01,
                orientation="h"
            ),
            height=500,
            margin=dict(t=100, l=50, r=50, b=50),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                title_font=dict(size=14),
                tickfont=dict(size=12),
                tickformat='%Y-%m-%d %H:%M',
                range=[
                    daily_counts[date_column].min(),
                    daily_counts[date_column].max()
                ]
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                title_font=dict(size=14),
                tickfont=dict(size=12)
            )
        )
        
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>" +
                         "Type: %{fullData.name}<br>" +
                         "Count: %{y}<extra></extra>",
            line=dict(width=2)
        )
        
        return fig
    
    def create_hotspot_chart(self, hotspot_data):
        """
        Create a bar chart showing top risk areas.

        Args:
            hotspot_data: DataFrame containing hotspot statistics

        Returns:
            Plotly figure object
        """
        fig = px.bar(
            hotspot_data,
            x='neighborhood',
            y='score',
            title='Top Risk Areas',
            labels={'score': 'Risk Score', 'neighborhood': 'Neighborhood'}
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=False
        )
        
        return fig
    
    def create_heatmap(self, data, radius=25):
        """
        Create a heatmap visualization of hotspots.

        Args:
            data: DataFrame containing hotspot data with center_lat, center_lon, and count
            radius: Radius of influence for each point in pixels

        Returns:
            Folium map object with heatmap layer
        """
        try:
            m = self.create_base_map()
            
            if data is not None and not data.empty:
                required_columns = ['center_lat', 'center_lon', 'count', 'density', 'area_km2']
                if not all(col in data.columns for col in required_columns):
                    st.error(f"Missing required columns. Expected: {required_columns}")
                    st.write("Available columns:", data.columns.tolist())
                    return m
                
                data = data.dropna(subset=['center_lat', 'center_lon', 'count', 'density'])
                
                if data.empty:
                    st.warning("No valid data points after removing NaN values")
                    return m
                
                data['center_lat'] = pd.to_numeric(data['center_lat'], errors='coerce')
                data['center_lon'] = pd.to_numeric(data['center_lon'], errors='coerce')
                data['count'] = pd.to_numeric(data['count'], errors='coerce')
                data['density'] = pd.to_numeric(data['density'], errors='coerce')
                data['density'] = data['density'].clip(0, 10000)
                
                data = data[
                    (data['center_lat'].between(40.4, 40.95)) &
                    (data['center_lon'].between(-74.25, -73.7))
                ]
                
                if data.empty:
                    st.warning("No valid coordinates within NYC bounds")
                    return m
                
                max_density = data['density'].max()
                if max_density > 0:
                    data['normalized_intensity'] = data['density'] / max_density
                else:
                    data['normalized_intensity'] = data['density']
                
                heat_data = []
                for _, row in data.iterrows():
                    if pd.notna(row['center_lat']) and pd.notna(row['center_lon']) and pd.notna(row['normalized_intensity']):
                        heat_data.append([
                            float(row['center_lat']), 
                            float(row['center_lon']), 
                            float(row['normalized_intensity'])
                        ])
                
                if heat_data:
                    HeatMap(
                        heat_data,
                        min_opacity=0.4,
                        radius=radius,
                        blur=15,
                        max_zoom=13,
                        gradient={
                            '0.0': '#fee5d9',
                            '0.2': '#fcae91',
                            '0.4': '#fb6a4a',
                            '0.6': '#de2d26',
                            '0.8': '#a50f15',
                            '1.0': '#67000d'
                        }
                    ).add_to(m)
                    
                    for _, row in data.iterrows():
                        size = min(20, max(8, (row['count'] / data['count'].max()) * 20))
                        
                        if row['density'] >= 1000:
                            density_str = f"{row['density']/1000:.1f}k"
                        elif row['density'] >= 100:
                            density_str = f"{row['density']:.0f}"
                        else:
                            density_str = f"{row['density']:.1f}"
                        
                        folium.CircleMarker(
                            location=[float(row['center_lat']), float(row['center_lon'])],
                            radius=size,
                            color='#67000d',
                            fill=True,
                            fillOpacity=0.7,
                            popup=folium.Popup(
                                f"<div style='font-family: Arial; font-size: 12px;'>"
                                f"<b>Risk Hotspot</b><br>"
                                f"Incidents: {int(row['count']):,}<br>"
                                f"Density: {density_str} incidents/km²<br>"
                                f"Area: {row['area_km2']:.2f} km²<br>"
                                f"Location: ({row['center_lat']:.4f}, {row['center_lon']:.4f})"
                                f"</div>",
                                max_width=300
                            ),
                            weight=2,
                            tooltip=f"Click for details ({int(row['count']):,} incidents)"
                        ).add_to(m)
            
                legend_html = """
                    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white;
                            padding: 10px; border-radius: 5px; box-shadow: 0 0 15px rgba(0,0,0,0.2);">
                        <h4 style="margin: 0 0 10px 0;">Risk Density</h4>
                        <div style="display: flex; align-items: center; margin: 5px 0;">
                            <div style="width: 20px; height: 20px; background: #67000d; margin-right: 10px;"></div>
                            <span>Very High (>1000/km²)</span>
                        </div>
                        <div style="display: flex; align-items: center; margin: 5px 0;">
                            <div style="width: 20px; height: 20px; background: #a50f15; margin-right: 10px;"></div>
                            <span>High (500-1000/km²)</span>
                        </div>
                        <div style="display: flex; align-items: center; margin: 5px 0;">
                            <div style="width: 20px; height: 20px; background: #de2d26; margin-right: 10px;"></div>
                            <span>Medium (100-500/km²)</span>
                        </div>
                        <div style="display: flex; align-items: center; margin: 5px 0;">
                            <div style="width: 20px; height: 20px; background: #fb6a4a; margin-right: 10px;"></div>
                            <span>Low (<100/km²)</span>
                        </div>
                    </div>
                """
                m.get_root().html.add_child(folium.Element(legend_html))
            
            folium.LayerControl(
                position='topright',
                collapsed=False
            ).add_to(m)
            
            return m
            
        except Exception as e:
            st.error(f"Error creating heatmap: {str(e)}")
            st.write("Debug information:")
            if data is not None:
                st.write("Data shape:", data.shape)
                st.write("Data columns:", data.columns.tolist())
                st.write("Data types:")
                st.write(data.dtypes)
                st.write("Sample data:")
                st.write(data.head())
                st.write("NaN counts:")
                st.write(data.isna().sum())
            return None
    
    def create_category_breakdown(self, df, category_column):
        """
        Create a pie chart showing breakdown by category.

        Args:
            df: DataFrame containing category data
            category_column: Name of the category column

        Returns:
            Plotly figure object
        """
        if df is None or df.empty:
            return None
            
        category_counts = df[category_column].value_counts()
        
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title=f'Breakdown by {category_column}'
        )
        
        fig.update_layout(height=400)
        return fig 