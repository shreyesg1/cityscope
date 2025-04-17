import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import streamlit as st
from datetime import datetime, timedelta

class RiskScorer:
    # Initialize scoring components
    def __init__(self):
        """
        Initialize the RiskScorer with data preprocessing components.

        Args:
            None

        Returns:
            None
        """
        self.scaler = StandardScaler()
    
    # Scale risk values
    def normalize_scores(self, scores):
        """
        Normalize risk scores to a 0-1 scale.

        Args:
            scores: Array of risk scores to normalize

        Returns:
            Normalized scores between 0 and 1
        """
        if scores is None or len(scores) == 0:
            return None
        return (scores - scores.min()) / (scores.max() - scores.min())
    
    # Find risk clusters
    def detect_hotspots(self, data, eps=None, min_samples=None):
        """
        Detect spatial clusters of high-risk areas using DBSCAN.

        Args:
            data: DataFrame containing latitude and longitude columns
            eps: Maximum distance between points in a cluster (if None, calculated based on data)
            min_samples: Minimum points required to form a cluster (if None, calculated based on data)

        Returns:
            DataFrame containing cluster statistics including location, count, area and density
        """
        try:
            # Convert lat/lon to meters
            coords = data[['latitude', 'longitude']].copy()
            lat_scale = 111000  # ~111km per degree of latitude
            lon_scale = 85000   # ~85km per degree of longitude at NYC's latitude
            coords['x'] = coords['longitude'] * lon_scale
            coords['y'] = coords['latitude'] * lat_scale
            
            # Scale coordinates for DBSCAN
            scaled_coords = self.scaler.fit_transform(coords[['x', 'y']])
            
            # Adjust clustering params based on data size
            if eps is None:
                eps = 500 * np.sqrt(10000 / len(data))
                eps = max(200, min(1000, eps))
            
            if min_samples is None:
                min_samples = max(5, int(np.log10(len(data)) * 3))
            
            # Run clustering
            eps_scaled = eps / np.sqrt(lat_scale * lon_scale)
            dbscan = DBSCAN(eps=eps_scaled, min_samples=min_samples)
            clusters = dbscan.fit_predict(scaled_coords)
            
            # Skip if no clusters found
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            if n_clusters == 0:
                return None
            
            # Calculate stats for each cluster
            data_with_clusters = data.copy()
            data_with_clusters['cluster'] = clusters
            
            cluster_stats = []
            for cluster_id in range(n_clusters):
                cluster_points = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
                center_lat = cluster_points['latitude'].mean()
                center_lon = cluster_points['longitude'].mean()
                
                # Get cluster boundaries
                min_lat, max_lat = cluster_points['latitude'].min(), cluster_points['latitude'].max()
                min_lon, max_lon = cluster_points['longitude'].min(), cluster_points['longitude'].max()
                
                area_km2 = max(0.01, (max_lat - min_lat) * 111 * (max_lon - min_lon) * 85)
                count = len(cluster_points)
                density = min(10000, count / area_km2) if area_km2 > 0 else 0
                
                cluster_stats.append({
                    'cluster_id': cluster_id,
                    'center_lat': center_lat,
                    'center_lon': center_lon,
                    'count': count,
                    'area_km2': area_km2,
                    'density': density
                })
            
            # Clean up
            result = pd.DataFrame(cluster_stats)
            result['density'] = pd.to_numeric(result['density'], errors='coerce').fillna(0)
            result['area_km2'] = pd.to_numeric(result['area_km2'], errors='coerce').fillna(0.01)
            
            result = result[
                (result['density'] > 0) & 
                (result['density'] < float('inf')) & 
                (result['area_km2'] > 0)
            ]
            
            return result
            
        except Exception as e:
            st.error(f"Error in hotspot detection: {str(e)}")
            return None
    
    def calculate_temporal_trend(self, data, date_column, window_days=7):
        """
        Calculate trends in incident frequency.

        Args:
            data: DataFrame containing incident data
            date_column: Name of the date column
            window_days: Rolling window size in days

        Returns:
            DataFrame with trend statistics
        """
        if data is None or len(data) == 0:
            return None
            
        daily_counts = data.groupby(data[date_column].dt.date).size()
        trend = daily_counts.rolling(window=window_days, min_periods=1).mean()
        return trend
    
    # Find growing risk areas
    def identify_emerging_risks(self, data, date_column, threshold_percentile=90):
        """
        Identify locations with rapidly increasing incident rates.

        Args:
            data: DataFrame containing incident data
            date_column: Name of the date column
            threshold_percentile: Percentile threshold for identifying high-risk areas

        Returns:
            DataFrame containing emerging risk locations
        """
        if data is None or len(data) == 0:
            return None
            
        recent_cutoff = datetime.now() - timedelta(days=7)
        recent_data = data[data[date_column] >= recent_cutoff]
        
        location_counts = recent_data.groupby(['latitude', 'longitude']).size()
        threshold = np.percentile(location_counts, threshold_percentile)
        high_risk_locations = location_counts[location_counts >= threshold]
        
        return pd.DataFrame(high_risk_locations)
    
    # Plan resource distribution
    def optimize_resource_allocation(self, hotspots, n_resources=10):
        """
        Optimize allocation of resources based on hotspot characteristics.

        Args:
            hotspots: DataFrame containing hotspot information
            n_resources: Number of resources to allocate

        Returns:
            DataFrame with resource allocation recommendations
        """
        if hotspots is None or len(hotspots) == 0:
            return None
            
        total_incidents = hotspots['count'].sum()
        allocations = (hotspots['count'] / total_incidents * n_resources).round()
        
        hotspots_with_allocation = hotspots.copy()
        hotspots_with_allocation['resources_allocated'] = allocations
        
        return hotspots_with_allocation 