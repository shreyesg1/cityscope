# Data cleaning and preprocessing module for NYC incident data
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import requests
import json

class DataProcessor:
    def __init__(self):
        """
        Initialize the DataProcessor with empty data sources.

        Args:
            None

        Returns:
            None
        """
        self.data_sources = {
            'crime': None,
            'complaints': None,
            'health': None,
            'infrastructure': None
        }
    
    def load_crime_data(self, file_path=None, api_url=None):
        """
        Load crime data from file or API.

        Args:
            file_path: Path to local CSV file containing crime data
            api_url: URL to fetch crime data from API

        Returns:
            DataFrame containing crime data
        """
        if file_path:
            self.data_sources['crime'] = pd.read_csv(file_path)
        elif api_url:
            response = requests.get(api_url)
            self.data_sources['crime'] = pd.DataFrame(response.json())
        return self.data_sources['crime']
    
    def load_complaints_data(self, file_path=None, api_url=None):
        """
        Load 311 complaints data from file or API.

        Args:
            file_path: Path to local CSV file containing complaints data
            api_url: URL to fetch complaints data from API

        Returns:
            DataFrame containing complaints data
        """
        if file_path:
            self.data_sources['complaints'] = pd.read_csv(file_path)
        elif api_url:
            response = requests.get(api_url)
            self.data_sources['complaints'] = pd.DataFrame(response.json())
        return self.data_sources['complaints']
    
    def clean_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate geographic coordinates
        """
        # Drop rows with missing coordinates
        df = df.dropna(subset=['latitude', 'longitude'])
        
        # Filter to NYC area bounds
        nyc_bounds = {
            'lat': (40.4774, 40.9176),  # NYC latitude bounds
            'lon': (-74.2591, -73.7002)  # NYC longitude bounds
        }
        
        mask = (
            (df['latitude'] >= nyc_bounds['lat'][0]) &
            (df['latitude'] <= nyc_bounds['lat'][1]) &
            (df['longitude'] >= nyc_bounds['lon'][0]) &
            (df['longitude'] <= nyc_bounds['lon'][1])
        )
        
        return df[mask].copy()

    def clean_dates(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Clean and standardize dates
        """
        # Convert to datetime
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Remove future dates and very old dates
        now = datetime.now()
        min_date = now - timedelta(days=365*2)  # 2 years ago
        
        mask = (
            (df[date_column] <= now) &
            (df[date_column] >= min_date)
        )
        
        return df[mask].copy()

    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[list] = None) -> pd.DataFrame:
        """
        Remove duplicate entries
        """
        if subset is None:
            return df.drop_duplicates()
        return df.drop_duplicates(subset=subset)

    def standardize_categories(self, df: pd.DataFrame, category_column: str) -> pd.DataFrame:
        """
        Standardize category names and group minor categories
        """
        # Convert to uppercase and strip whitespace
        df[category_column] = df[category_column].str.upper().str.strip()
        
        # Group categories with few occurrences
        value_counts = df[category_column].value_counts()
        min_count = max(10, len(df) * 0.01)  # At least 1% of total (10 min)
        
        major_categories = value_counts[value_counts >= min_count].index
        df.loc[~df[category_column].isin(major_categories), category_column] = 'OTHER'
        
        return df

    def process_dataset(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Apply all processing steps to a dataset
        """
        if df is None or df.empty:
            return df
            
        # Apply each cleaning step
        if 'date_column' in config:
            df = self.clean_dates(df, config['date_column'])
            
        df = self.clean_coordinates(df)
        
        if 'category_column' in config:
            df = self.standardize_categories(df, config['category_column'])
            
        if 'dedup_columns' in config:
            df = self.remove_duplicates(df, config['dedup_columns'])
        else:
            df = self.remove_duplicates(df)
            
        return df
    
    def clean_data(self, df, data_type):
        """
        Clean and standardize data.

        Args:
            df: DataFrame to clean
            data_type: Type of data being cleaned (crime, complaints, etc.)

        Returns:
            Cleaned DataFrame with standardized columns
        """
        if df is None:
            return None
            
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])
        
        df = df.fillna(method='ffill')
        
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df = self.clean_coordinates(df)
        
        return df
    
    def calculate_risk_score(self, weights):
        """
        Calculate composite risk score based on weights.

        Args:
            weights: Dictionary mapping data types to their weights in risk calculation

        Returns:
            Series containing risk scores by neighborhood
        """
        scores = {}
        
        for data_type, weight in weights.items():
            if self.data_sources[data_type] is not None:
                df = self.clean_data(self.data_sources[data_type], data_type)
                if df is not None:
                    scores[data_type] = df.groupby('neighborhood').size() * weight

        if scores:
            combined_score = pd.concat(scores.values(), axis=1).sum(axis=1)
            return combined_score
        return None
    
    def get_trend_data(self, start_date, end_date):
        """
        Get trend data for the specified date range.

        Args:
            start_date: Start date for trend analysis
            end_date: End date for trend analysis

        Returns:
            Dictionary mapping data types to their trend data
        """
        trend_data = {}
        
        for data_type, df in self.data_sources.items():
            if df is not None:
                df = self.clean_data(df, data_type)
                if df is not None and 'date' in df.columns:
                    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
                    trend_data[data_type] = df[mask].groupby('date').size()
        
        return trend_data 