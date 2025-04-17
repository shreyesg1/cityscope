import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import streamlit as st
import os
import time
from functools import wraps
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, Optional, List, Tuple

def create_requests_session():
    """
    Create a requests session with retry strategy.

    Args:
        None

    Returns:
        Session object configured with retry strategy
    """
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[408, 429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def rate_limit(seconds=1):
    """
    Decorator to rate limit API calls.

    Args:
        seconds: Number of seconds to wait between calls

    Returns:
        Decorated function with rate limiting
    """
    def decorator(func):
        last_called = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            if func.__name__ not in last_called:
                last_called[func.__name__] = now
            else:
                elapsed = now - last_called[func.__name__]
                if elapsed < seconds:
                    time.sleep(seconds - elapsed)
            last_called[func.__name__] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

@st.cache_data(ttl=3600)
def fetch_data_from_api(endpoint, params, headers=None):
    """
    Generic function to fetch data from API with caching.

    Args:
        endpoint: API endpoint URL
        params: Query parameters for the API request
        headers: Optional headers for the API request

    Returns:
        JSON response from the API or None if request fails
    """
    session = create_requests_session()
    
    try:
        with st.spinner(f"Fetching data from {endpoint.split('/')[-2]}..."):
            response = session.get(
                endpoint,
                headers=headers,
                params=params,
                timeout=(10, 60)
            )
            response.raise_for_status()
            return response.json()
    except requests.exceptions.Timeout:
        st.error(f"Request to {endpoint.split('/')[-2]} timed out.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"API Request failed: {str(e)}")
        return None
    finally:
        session.close()

# Handles API calls to NYC Open Data
class DataFetcher:
    def __init__(self, config_path='config.json'):
        """
        Initialize data fetcher with configuration.

        Args:
            config_path: Path to configuration file

        Returns:
            None
        """
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            self.api_endpoints = self.config['api_endpoints']
            self.map_center = self.config['map_settings']['center']
            
            self.session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
            
            self.app_token = os.environ.get('NYC_OPEN_DATA_TOKEN')
            if not self.app_token:
                try:
                    self.app_token = st.secrets.get("nyc_open_data_token")
                except:
                    self.app_token = None
            
        except Exception as e:
            st.error("Error initializing data fetcher")
            raise

    def _make_api_request(self, url, params=None):
        """
        Make API request with error handling and app token.

        Args:
            url: API endpoint URL
            params: Query parameters for the request

        Returns:
            JSON response data or None if request fails
        """
        try:
            headers = {}
            if self.app_token:
                headers['X-App-Token'] = self.app_token
            
            response = self.session.get(url, params=params, headers=headers, timeout=30)
            
            if response.status_code != 200:
                return None
                
            response.raise_for_status()
            
            data = response.json()
            if not data:
                return None
                
            return data
            
        except requests.exceptions.Timeout:
            return None
        except requests.exceptions.ConnectionError:
            return None
        except requests.exceptions.HTTPError:
            return None
        except json.JSONDecodeError:
            return None
        except Exception:
            return None

    def fetch_311_data(self, days=30):
        """
        Fetch recent 311 complaints with fallback to sample data.

        Args:
            days: Number of days of historical data to fetch

        Returns:
            DataFrame containing 311 complaints or sample data if fetch fails
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        all_data = []
        offset = 0
        
        try:
            while True:
                params = {
                    '$where': f"created_date between '{start_date.strftime('%Y-%m-%dT00:00:00')}' and '{end_date.strftime('%Y-%m-%dT23:59:59')}'",
                    '$select': 'created_date,complaint_type,descriptor,latitude,longitude,incident_zip,borough',
                    '$limit': 50000,
                    '$offset': offset,
                    '$order': 'created_date DESC'
                }
                
                data = self._make_api_request(self.api_endpoints['nyc_311'], params)
                
                if not data:
                    if offset == 0:
                        st.error("Failed to fetch 311 data")
                    break
                
                all_data.extend(data)
                
                if len(data) < 50000:
                    break
                
                offset += 50000
                time.sleep(0.1)
            
            if all_data:
                df = pd.DataFrame(all_data)
                if not df.empty and 'latitude' in df.columns and 'longitude' in df.columns:
                    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
                    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
                    df['created_date'] = pd.to_datetime(df['created_date'])
                    df = df.dropna(subset=['latitude', 'longitude'])
                    return df
            
            st.error("No valid 311 data received")
            return pd.DataFrame()
            
        except Exception as e:
            st.error(f"Error fetching 311 data: {str(e)}")
            return pd.DataFrame()

    def fetch_crime_data(self, days=30):
        """
        Fetch recent NYPD complaint data with fallback to sample data.

        Args:
            days: Number of days of historical data to fetch

        Returns:
            DataFrame containing crime data or sample data if fetch fails
        """
        try:
            test_params = {
                '$select': '*',
                '$limit': 1
            }
            
            response = self.session.get(
                self.api_endpoints['nypd_complaints'],
                params=test_params,
                headers={'X-App-Token': self.app_token} if self.app_token else {},
                timeout=30
            )
            
            if response.status_code != 200:
                st.error("NYPD API test failed")
                return pd.DataFrame()
            
            test_data = response.json()
            if not test_data:
                st.error("NYPD API test returned no data")
                return pd.DataFrame()
            
            date_field = None
            possible_date_fields = ['cmplnt_fr_dt', 'complaint_date', 'date', 'rpt_dt']
            for field in possible_date_fields:
                if field in test_data[0]:
                    date_field = field
                    break
            
            if not date_field:
                st.error("Date field not found in NYPD data")
                return pd.DataFrame()
            
            most_recent = pd.to_datetime(test_data[0][date_field])
            end_date = most_recent
            start_date = end_date - timedelta(days=days)
            
            start_str = start_date.strftime('%Y-%m-%dT00:00:00.000')
            end_str = end_date.strftime('%Y-%m-%dT23:59:59.999')
            
            all_data = []
            offset = 0
            batch_size = 10000
            
            while True:
                params = {
                    '$select': '*',
                    '$where': f"{date_field} >= '{start_str}' AND {date_field} <= '{end_str}'",
                    '$limit': batch_size,
                    '$offset': offset
                }
                
                try:
                    response = self.session.get(
                        self.api_endpoints['nypd_complaints'],
                        params=params,
                        headers={'X-App-Token': self.app_token} if self.app_token else {},
                        timeout=30
                    )
                    
                    if response.status_code != 200:
                        break
                    
                    batch_data = response.json()
                    if not batch_data:
                        break
                    
                    all_data.extend(batch_data)
                    
                    if len(batch_data) < batch_size:
                        break
                    
                    offset += batch_size
                    time.sleep(0.1)
                    
                except Exception:
                    if len(all_data) == 0:
                        st.error("Failed to fetch any NYPD data")
                        return pd.DataFrame()
                    break
            
            if not all_data:
                st.error("No NYPD data received")
                return pd.DataFrame()
            
            df = pd.DataFrame(all_data)
            
            df[date_field] = pd.to_datetime(df[date_field])
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            
            df = df.dropna(subset=['latitude', 'longitude'])
            df = df.sort_values(date_field, ascending=False)
            
            if date_field != 'cmplnt_fr_dt':
                df = df.rename(columns={date_field: 'cmplnt_fr_dt'})
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching NYPD data: {str(e)}")
            return pd.DataFrame()

    def fetch_health_data(self, days=30):
        """
        Fetch recent restaurant inspection data with fallback to sample data.

        Args:
            days: Number of days of historical data to fetch

        Returns:
            DataFrame containing health inspection data or sample data if fetch fails
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        params = {
            '$where': f"inspection_date between '{start_date.strftime('%Y-%m-%d')}' and '{end_date.strftime('%Y-%m-%d')}'",
            '$select': 'inspection_date,violation_code,violation_description,score,latitude,longitude,boro',
            '$limit': 10000,
            '$order': 'inspection_date DESC'
        }
        
        data = self._make_api_request(self.api_endpoints['restaurant_inspections'], params)
        
        if data:
            df = pd.DataFrame(data)
            if not df.empty and 'latitude' in df.columns and 'longitude' in df.columns:
                df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
                df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
                df['inspection_date'] = pd.to_datetime(df['inspection_date'])
                df['score'] = pd.to_numeric(df['score'], errors='coerce')
                df = df.dropna(subset=['latitude', 'longitude'])
                return df
        
        st.error("No valid health inspection data received")
        return pd.DataFrame()

    def get_all_data(self, days=30):
        """
        Fetch all datasets (311 complaints, crime reports, and health violations).
        
        Args:
            days: Number of days of historical data to fetch
            
        Returns:
            Dictionary containing the three datasets
        """
        try:
            progress_text = "Loading data..."
            progress_bar = st.progress(0, text=progress_text)
            
            complaints_df = self.fetch_311_data(days)
            progress_bar.progress(0.33, text=f"Loading... ({len(complaints_df):,} incidents)")
            
            crime_df = self.fetch_crime_data(days)
            total_incidents = len(complaints_df) + len(crime_df)
            progress_bar.progress(0.66, text=f"Loading... ({total_incidents:,} incidents)")
            
            health_df = self.fetch_health_data(days)
            total_incidents += len(health_df)
            progress_bar.progress(1.0, text=f"Loaded {total_incidents:,} incidents")
            
            return {
                'complaints': complaints_df,
                'crime': crime_df,
                'health': health_df
            }
            
        except Exception as e:
            st.error("Error loading data")
            return None 