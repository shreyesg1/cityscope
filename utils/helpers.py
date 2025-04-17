import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, Optional

# Load app settings
def load_config(config_path: str = 'config.json') -> Dict:
    """Load configuration settings"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {}

# Save app settings
def save_config(config, config_path='config.json'):
    """Save configuration file"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

# Check date inputs
def validate_date_range(start_date, end_date):
    """Validate date range inputs"""
    if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
        raise ValueError("Dates must be datetime objects")
    if start_date > end_date:
        raise ValueError("Start date must be before end date")
    return start_date, end_date

# Format score display
def format_risk_score(score: float) -> str:
    """Format risk score"""
    if score is None:
        return "N/A"
    return f"{score:.2f}"

# Calculate value changes
def calculate_percent_change(current, previous):
    """Calculate percentage difference"""
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100

# Format change display
def format_percent_change(change):
    """Format change percentage"""
    if change > 0:
        return f"+{change:.1f}%"
    return f"{change:.1f}%"

# Find location column
def get_location_column(data):
    """Find location field name"""
    location_columns = {
        'neighborhood': 'neighborhood',
        'borough': 'borough',
        'boro': 'boro',
        'boro_nm': 'boro_nm',
        'incident_zip': 'incident_zip',
        'zip_code': 'zip_code'
    }
    
    for col in location_columns:
        if col in data.columns:
            return col
    return None

# Generate data stats
def generate_summary_statistics(df: pd.DataFrame) -> Dict:
    """Calculate dataset statistics"""
    if df is None or df.empty:
        return {
            'total_incidents': 0,
            'areas_affected': 0,
            'average_risk_score': None
        }
    
    # Basic counts
    total_incidents = len(df)
    
    # Count unique locations
    if 'latitude' in df.columns and 'longitude' in df.columns:
        locations = df.groupby(['latitude', 'longitude']).size()
        areas_affected = len(locations)
    else:
        areas_affected = 0
    
    # Calculate risk score if possible
    if 'risk_score' in df.columns:
        avg_risk = df['risk_score'].mean()
    else:
        avg_risk = None
    
    return {
        'total_incidents': total_incidents,
        'areas_affected': areas_affected,
        'average_risk_score': avg_risk
    } 