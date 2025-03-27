#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os
import sys
import shap
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Add the src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import AttritionPredictionModel
from src.model import AttritionPredictionModel
from src.recommendation_engine import generate_recommendations, identify_risk_clusters

# Set page configuration
st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="👥",
    page_title="Employee Attrition Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with modern design
# Custom CSS with modern design
st.markdown("""
<style>
    /* Modern Typography */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Main Layout */
    .main {
        background-color: #f8fafc;
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 1rem;
        background-color: #f8fafc;
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    /* Headers */
    .main-header {
        font-family: 'Poppins', sans-serif;
        font-size: 2.5rem;
        font-family: 'Poppins', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        color: #1e293b;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }
    
    
    
    .sub-header {
        font-family: 'Poppins', sans-serif;
        font-size: 1.5rem;
        font-family: 'Poppins', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #334155;
        color: #334155;
        margin-bottom: 1.5rem;
        letter-spacing: -0.01em;
    }
    
    /* Cards */
    .stCard {
        background-color: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
        max-width: 100%;
    /* Cards */
    .stCard {
        background-color: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
        max-width: 100%;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 1rem;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        transition: transform 0.2s;
        max-width: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        transition: transform 0.2s;
        max-width: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-family: 'Poppins', sans-serif;
        font-family: 'Poppins', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-family: 'Poppins', sans-serif;
        font-family: 'Poppins', sans-serif;
        font-size: 0.875rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Risk Indicators */
    .risk-high {
        color: #dc2626;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        background-color: #fee2e2;
    /* Risk Indicators */
    .risk-high {
        color: #dc2626;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        background-color: #fee2e2;
    }
    
    
    .risk-medium {
        color: #f59e0b;
    
    .risk-medium {
        color: #f59e0b;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        background-color: #fef3c7;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        background-color: #fef3c7;
    }
    
    
    
    .risk-low {
        color: #059669;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        background-color: #d1fae5;
    .risk-low {
        color: #059669;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        background-color: #d1fae5;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: transparent;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Poppins', sans-serif;
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        color: #64748b;
        padding: 0.75rem 1.5rem;
        border-radius: 0.75rem;
        padding: 0.75rem 1.5rem;
        border-radius: 0.75rem;
        transition: all 0.2s;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        background-color: white;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1e293b;
        background-color: #1e293b;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons */
    .stButton button {
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        background-color: #1e293b;
        color: white;
        border-radius: 0.75rem;
        padding: 0.75rem 1.5rem;
        transition: all 0.2s;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .stButton button:hover {
        background-color: #334155;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Select Boxes */
    .stSelectbox select {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        color: #1e293b;
        padding: 0.75rem;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background-color: #1e293b;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons */
    .stButton button {
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        background-color: #1e293b;
        color: white;
        border-radius: 0.75rem;
        padding: 0.75rem 1.5rem;
        transition: all 0.2s;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .stButton button:hover {
        background-color: #334155;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Select Boxes */
    .stSelectbox select {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        color: #1e293b;
        padding: 0.75rem;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background-color: #1e293b;
    }
    
    /* Data Tables */
    .dataframe {
        font-family: 'Poppins', sans-serif;
        border-radius: 1rem;
        font-family: 'Poppins', sans-serif;
        border-radius: 1rem;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        background-color: white;
        max-width: 100%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        background-color: white;
        max-width: 100%;
    }
    
    .dataframe thead th {
        background-color: #f8fafc;
        font-weight: 600;
        color: #1e293b;
        color: #1e293b;
        padding: 1rem;
    }
    
    .dataframe tbody td {
        padding: 1rem;
        color: #64748b;
        background-color: white;
        background-color: white;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: #f8fafc;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: white;
        background-color: white;
        border-right: 1px solid #e2e8f0;
    }
    
    .sidebar .sidebar-content {
        padding: 2rem;
        padding: 2rem;
    }
    
    /* Charts */
    .js-plotly-plot {
        border-radius: 1rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        background-color: white !important;
        max-width: 100% !important;
    /* Charts */
    .js-plotly-plot {
        border-radius: 1rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        background-color: white !important;
        max-width: 100% !important;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main {
            padding: 0 0.5rem;
        }
        
        .main {
            padding: 0 0.5rem;
        }
        
        .main-header {
            font-size: 2rem;
        }
        
        .sub-header {
            font-size: 1.25rem;
            font-size: 1.25rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def preprocess_data(df, features):
    """
    Preprocess the data by handling missing values and scaling features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        features (list): List of features to preprocess
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Separate numerical and categorical features
    numerical_features = [f for f in features if not (f.startswith('Business_Unit_') or 
                                                   f.startswith('Region_') or 
                                                   f in ['bu', 'region'])]
    categorical_features = [f for f in features if f.startswith('Business_Unit_') or 
                                                 f.startswith('Region_') or 
                                                 f in ['bu', 'region']]
    
    # Handle missing values in numerical features
    if numerical_features:
        imputer = SimpleImputer(strategy='mean')
        df_processed[numerical_features] = imputer.fit_transform(df_processed[numerical_features])
    
    # Handle missing values in categorical features
    for cat_feature in categorical_features:
        if cat_feature in df_processed.columns:
            df_processed[cat_feature] = df_processed[cat_feature].fillna('Unknown')
    
    # Scale numerical features
    if numerical_features:
        scaler = StandardScaler()
        df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])
    
    return df_processed

def validate_data(df, features):
    """
    Validate the data and check for potential issues.
    
    Args:
        df (pd.DataFrame): Input dataframe
        features (list): List of features to validate
        
    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    # Separate numerical and categorical features
    numerical_features = [f for f in features if not (f.startswith('Business_Unit_') or 
                                                   f.startswith('Region_') or 
                                                   f in ['bu', 'region'])]
    categorical_features = [f for f in features if f.startswith('Business_Unit_') or 
                                                 f.startswith('Region_') or 
                                                 f in ['bu', 'region']]
    
    # Check for missing values in numerical features
    if numerical_features:
        missing_values = df[numerical_features].isnull().sum()
        if missing_values.any():
            return False, f"Missing values found in numerical features: {missing_values[missing_values > 0].to_dict()}"
    
    # Check for infinite values in numerical features
    if numerical_features:
        # Convert to numeric, coercing errors to NaN
        numeric_df = df[numerical_features].apply(pd.to_numeric, errors='coerce')
        inf_values = np.isinf(numeric_df).sum()
        if inf_values.any():
            return False, f"Infinite values found in numerical features: {inf_values[inf_values > 0].to_dict()}"
    
    # Check for constant columns in numerical features only
    if numerical_features:
        constant_cols = df[numerical_features].columns[df[numerical_features].nunique() == 1]
        if len(constant_cols) > 0:
            # Log constant columns for debugging
            print("Constant columns found:", constant_cols.tolist())
            print("Values of constant columns:")
            for col in constant_cols:
                print(f"{col}: {df[col].iloc[0]}")
            return False, f"Constant columns found in numerical features: {constant_cols.tolist()}"
    
    return True, "Data validation successful"

    }
</style>
""", unsafe_allow_html=True)

def preprocess_data(df, features):
    """
    Preprocess the data by handling missing values and scaling features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        features (list): List of features to preprocess
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Separate numerical and categorical features
    numerical_features = [f for f in features if not (f.startswith('Business_Unit_') or 
                                                   f.startswith('Region_') or 
                                                   f in ['bu', 'region'])]
    categorical_features = [f for f in features if f.startswith('Business_Unit_') or 
                                                 f.startswith('Region_') or 
                                                 f in ['bu', 'region']]
    
    # Handle missing values in numerical features
    if numerical_features:
        imputer = SimpleImputer(strategy='mean')
        df_processed[numerical_features] = imputer.fit_transform(df_processed[numerical_features])
    
    # Handle missing values in categorical features
    for cat_feature in categorical_features:
        if cat_feature in df_processed.columns:
            df_processed[cat_feature] = df_processed[cat_feature].fillna('Unknown')
    
    # Scale numerical features
    if numerical_features:
        scaler = StandardScaler()
        df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])
    
    return df_processed

def validate_data(df, features):
    """
    Validate the data and check for potential issues.
    
    Args:
        df (pd.DataFrame): Input dataframe
        features (list): List of features to validate
        
    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    # Separate numerical and categorical features
    numerical_features = [f for f in features if not (f.startswith('Business_Unit_') or 
                                                   f.startswith('Region_') or 
                                                   f in ['bu', 'region'])]
    categorical_features = [f for f in features if f.startswith('Business_Unit_') or 
                                                 f.startswith('Region_') or 
                                                 f in ['bu', 'region']]
    
    # Check for missing values in numerical features
    if numerical_features:
        missing_values = df[numerical_features].isnull().sum()
        if missing_values.any():
            return False, f"Missing values found in numerical features: {missing_values[missing_values > 0].to_dict()}"
    
    # Check for infinite values in numerical features
    if numerical_features:
        # Convert to numeric, coercing errors to NaN
        numeric_df = df[numerical_features].apply(pd.to_numeric, errors='coerce')
        inf_values = np.isinf(numeric_df).sum()
        if inf_values.any():
            return False, f"Infinite values found in numerical features: {inf_values[inf_values > 0].to_dict()}"
    
    # Check for constant columns in numerical features only
    if numerical_features:
        constant_cols = df[numerical_features].columns[df[numerical_features].nunique() == 1]
        if len(constant_cols) > 0:
            # Log constant columns for debugging
            print("Constant columns found:", constant_cols.tolist())
            print("Values of constant columns:")
            for col in constant_cols:
                print(f"{col}: {df[col].iloc[0]}")
            return False, f"Constant columns found in numerical features: {constant_cols.tolist()}"
    
    return True, "Data validation successful"

@st.cache_data
def load_data():
    """Load and cache the preprocessed employee data"""
    """Load and cache the preprocessed employee data"""
    try:
        df = pd.read_csv('data/preprocessed_employee_data.csv')
        df = pd.read_csv('data/preprocessed_employee_data.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_prediction_resources():
    """Load and cache the model and explainer"""
    """Load and cache the model and explainer"""
    try:
        model = AttritionPredictionModel.load_model()
        explainer = shap.TreeExplainer(model)
        return model, explainer
    except Exception as e:
        st.error(f"Error loading prediction resources: {str(e)}")
        return None, None
        model = AttritionPredictionModel.load_model()
        explainer = shap.TreeExplainer(model)
        return model, explainer
    except Exception as e:
        st.error(f"Error loading prediction resources: {str(e)}")
        return None, None

def format_risk_score(score):
    """Format the risk score with appropriate styling"""
    # Convert to percentage (0-100) and apply consistent adjustment
    adjusted_score = min(score * 1.5, 1.0)  # Apply the same adjustment factor
    percentage = adjusted_score * 100
    
    if percentage >= 70:  # High risk threshold
        return f'<span class="risk-high">High Risk ({percentage:.0f}%)</span>'
    elif percentage >= 50:  # Medium risk threshold
        return f'<span class="risk-medium">Medium Risk ({percentage:.0f}%)</span>'
    else:  # Low risk
        return f'<span class="risk-low">Low Risk ({percentage:.0f}%)</span>'

def predict_all_employees(df, model, explainer, features):
    """Generate predictions for all employees"""
    try:
        # Define the exact feature names expected by the model (23 features)
        required_features = [
            'tenure', 'performance_score', 'last_promotion', 'training_hours', 'role_changes',
            'survey_satisfaction', 'peer_attrition_last60d', 'leave_days_taken', 'skill_relevance',
            'team_attrition_rate', 'dept_attrition_rate', 'manager_avg_performance', 'manager_team_size',
            'promotion_delay', 'never_promoted', 'engagement_score', 'stagnation_risk',
            'performance_above_dept_avg', 'leave_utilization', 'high_performer_no_promo',
            'Business_Unit_BU1', 'Business_Unit_BU2', 'Business_Unit_BU3'
        ]
        
        # Ensure all required features are present
        missing_features = set(required_features) - set(df.columns)
        if missing_features:
            st.error(f"Missing required features: {missing_features}")
            return None, None
        
        # Use only the required features for prediction
        X = df[required_features]
        
        # Generate predictions
        probabilities = model.predict_proba(X)[:, 1]
        
        # Generate SHAP explanations
        explanations = explainer.shap_values(X)
        if isinstance(explanations, list):
            explanations = explanations[0]  # For binary classification, get first class
            
        # Create a full explanations array with zeros for categorical features
        full_explanations = np.zeros((len(df), len(features)))
        for i, feature in enumerate(features):
            if feature in required_features:
                feature_idx = required_features.index(feature)
                full_explanations[:, i] = explanations[:, feature_idx]
        
        return probabilities, full_explanations
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
        return None, None
        if isinstance(explanations, list):
            explanations = explanations[0]  # For binary classification, get first class
            
        # Create a full explanations array with zeros for categorical features
        full_explanations = np.zeros((len(df), len(features)))
        for i, feature in enumerate(features):
            if feature in required_features:
                feature_idx = required_features.index(feature)
                full_explanations[:, i] = explanations[:, feature_idx]
        
        return probabilities, full_explanations
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
        return None, None

def display_overview_tab(df, transformed_df, probabilities, features, explanations):
    """Display the overview tab with key metrics and visualizations"""
    st.markdown('<h1 class="main-header">Employee Attrition Analytics</h1>', unsafe_allow_html=True)
def display_overview_tab(df, transformed_df, probabilities, features, explanations):
    """Display the overview tab with key metrics and visualizations"""
    st.markdown('<h1 class="main-header">Employee Attrition Analytics</h1>', unsafe_allow_html=True)
    
    # Employee Search
    st.markdown('<h2 class="sub-header">Search Employee</h2>', unsafe_allow_html=True)
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_id = st.text_input("Enter Employee ID", placeholder="e.g., EID770487", key="search_input")
        if search_id and not search_id.startswith("EID"):
            search_id = f"EID{search_id}"
    with search_col2:
        if st.button("Search", key="search_btn"):
            if search_id in df.index:
                st.session_state.selected_employee = search_id
                st.session_state.active_tab = 1  # Switch to Employee Analysis tab
                st.experimental_rerun()
            else:
                st.error("Employee ID not found")
    
    # Initialize risk filter in session state if not exists
    if 'risk_filter' not in st.session_state:
        st.session_state.risk_filter = 'all'
    
    # Use raw probabilities without scaling
    adjusted_probabilities = probabilities
    
    # Key Metrics with clickable cards
    col1, col2, col3, col4 = st.columns(4)
    
    
    with col1:
        if st.button("All Employees", key="all_btn"):
            st.session_state.risk_filter = 'all'
            st.experimental_rerun()
        if st.button("All Employees", key="all_btn"):
            st.session_state.risk_filter = 'all'
            st.experimental_rerun()
        st.markdown("""
            <div class="metric-card" style="cursor: pointer;">
            <div class="metric-card" style="cursor: pointer;">
                <div class="metric-value">{}</div>
                <div class="metric-label">All Employees</div>
                <div class="metric-label">All Employees</div>
            </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        high_risk = (adjusted_probabilities >= 0.70).sum()
        if st.button("High Risk", key="high_risk_btn"):
            st.session_state.risk_filter = 'high'
            st.experimental_rerun()
        high_risk = (adjusted_probabilities >= 0.70).sum()
        if st.button("High Risk", key="high_risk_btn"):
            st.session_state.risk_filter = 'high'
            st.experimental_rerun()
        st.markdown("""
            <div class="metric-card" style="cursor: pointer;">
            <div class="metric-card" style="cursor: pointer;">
                <div class="metric-value">{}</div>
                <div class="metric-label">High Risk Employees</div>
            </div>
        """.format(high_risk), unsafe_allow_html=True)
        """.format(high_risk), unsafe_allow_html=True)
    
    with col3:
        medium_risk = ((adjusted_probabilities >= 0.50) & (adjusted_probabilities < 0.70)).sum()
        if st.button("Medium Risk", key="medium_risk_btn"):
            st.session_state.risk_filter = 'medium'
            st.experimental_rerun()
        medium_risk = ((adjusted_probabilities >= 0.50) & (adjusted_probabilities < 0.70)).sum()
        if st.button("Medium Risk", key="medium_risk_btn"):
            st.session_state.risk_filter = 'medium'
            st.experimental_rerun()
        st.markdown("""
            <div class="metric-card" style="cursor: pointer;">
            <div class="metric-card" style="cursor: pointer;">
                <div class="metric-value">{}</div>
                <div class="metric-label">Medium Risk Employees</div>
            </div>
        """.format(medium_risk), unsafe_allow_html=True)
    
    with col4:
        low_risk = (adjusted_probabilities < 0.50).sum()
        if st.button("Low Risk", key="low_risk_btn"):
            st.session_state.risk_filter = 'low'
            st.experimental_rerun()
        low_risk = (adjusted_probabilities < 0.50).sum()
        if st.button("Low Risk", key="low_risk_btn"):
            st.session_state.risk_filter = 'low'
            st.experimental_rerun()
        st.markdown("""
            <div class="metric-card" style="cursor: pointer;">
            <div class="metric-card" style="cursor: pointer;">
                <div class="metric-value">{}</div>
                <div class="metric-label">Low Risk Employees</div>
            </div>
        """.format(low_risk), unsafe_allow_html=True)
    
    # Filter data based on selected risk level
    if st.session_state.risk_filter == 'high':
        filtered_mask = adjusted_probabilities >= 0.70
    elif st.session_state.risk_filter == 'medium':
        filtered_mask = (adjusted_probabilities >= 0.50) & (adjusted_probabilities < 0.70)
    elif st.session_state.risk_filter == 'low':
        filtered_mask = adjusted_probabilities < 0.50
    else:  # 'all'
        filtered_mask = np.ones(len(df), dtype=bool)
    
    filtered_df = df[filtered_mask]
    filtered_probabilities = adjusted_probabilities[filtered_mask]
    filtered_transformed_df = transformed_df[filtered_mask]
    
    # Business Unit and Region Heatmaps
    st.markdown('<h2 class="sub-header">Attrition Heatmaps</h2>', unsafe_allow_html=True)
    # Filter data based on selected risk level
    if st.session_state.risk_filter == 'high':
        filtered_mask = adjusted_probabilities >= 0.70
    elif st.session_state.risk_filter == 'medium':
        filtered_mask = (adjusted_probabilities >= 0.50) & (adjusted_probabilities < 0.70)
    elif st.session_state.risk_filter == 'low':
        filtered_mask = adjusted_probabilities < 0.50
    else:  # 'all'
        filtered_mask = np.ones(len(df), dtype=bool)
    
    filtered_df = df[filtered_mask]
    filtered_probabilities = adjusted_probabilities[filtered_mask]
    filtered_transformed_df = transformed_df[filtered_mask]
    
    # Business Unit and Region Heatmaps
    st.markdown('<h2 class="sub-header">Attrition Heatmaps</h2>', unsafe_allow_html=True)
    
    # Create tabs for different heatmap views
    heatmap_tab1, heatmap_tab2 = st.tabs(["Business Unit Heatmap", "Region Heatmap"])
    
    with heatmap_tab1:
        # Business Unit Heatmap
        # Get all Business Unit columns
        bu_columns = [col for col in df.columns if col.startswith('Business_Unit_')]
        if bu_columns:
            # Calculate attrition rates by Business Unit
            bu_attrition = pd.DataFrame()
            
            for bu_col in bu_columns:
                bu_name = bu_col.replace('Business_Unit_', '')
                # Filter for employees in this BU
                bu_employees = df[df[bu_col] == 1]
                if not bu_employees.empty:
                    # Calculate overall attrition rate for this BU
                    bu_attrition[bu_name] = [bu_employees['AttritionLabel'].mean()]
            
            # Create heatmap with improved colors
            fig_bu = px.imshow(
                bu_attrition,
                title="Attrition Rate by Business Unit",
                labels=dict(x="Business Unit", y="Attrition Rate", color="Attrition Rate"),
                aspect="auto",
                color_continuous_scale=['#2ecc71', '#f1c40f', '#e74c3c']  # Green to Yellow to Red
            )
            
            # Update layout
            fig_bu.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Poppins'),
                xaxis_title="Business Unit",
                yaxis_title="Attrition Rate",
                coloraxis_colorbar_title="Attrition Rate"
            )
            
            # Add hover template
            fig_bu.update_traces(
                hovertemplate="<b>%{x}</b><br>" +
                            "Attrition Rate: <b>%{z:.1%}</b><br>" +
                            "<extra></extra>"
            )
            
            st.plotly_chart(fig_bu, use_container_width=True)
            
            # Add summary statistics
            st.markdown("""
                <div style='background-color: #f8fafc; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                    <h4>Business Unit Summary</h4>
                    <p>Hover over the heatmap to see detailed attrition rates for each Business Unit.</p>
                    <p>The color intensity indicates the attrition rate:</p>
                    <ul>
                        <li><span style='color: #e74c3c;'>Red</span> - High attrition (>30%)</li>
                        <li><span style='color: #f1c40f;'>Yellow</span> - Medium attrition (15-30%)</li>
                        <li><span style='color: #2ecc71;'>Green</span> - Low attrition (<15%)</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Business Unit data is not available in the current dataset.")
    
    with heatmap_tab2:
        # Region Heatmap
        # Get all Region columns
        region_columns = [col for col in df.columns if col.startswith('Region_')]
        if region_columns:
            # Calculate attrition rates by Region
            region_attrition = pd.DataFrame()
            
            for region_col in region_columns:
                region_name = region_col.replace('Region_', '')
                # Filter for employees in this Region
                region_employees = df[df[region_col] == 1]
                if not region_employees.empty:
                    # Calculate overall attrition rate for this Region
                    region_attrition[region_name] = [region_employees['AttritionLabel'].mean()]
            
            # Create heatmap with improved colors
            fig_region = px.imshow(
                region_attrition,
                title="Attrition Rate by Region",
                labels=dict(x="Region", y="Attrition Rate", color="Attrition Rate"),
                aspect="auto",
                color_continuous_scale=['#2ecc71', '#f1c40f', '#e74c3c']  # Green to Yellow to Red
            )
            
            # Update layout
            fig_region.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Poppins'),
                xaxis_title="Region",
                yaxis_title="Attrition Rate",
                coloraxis_colorbar_title="Attrition Rate"
            )
            
            # Add hover template
            fig_region.update_traces(
                hovertemplate="<b>%{x}</b><br>" +
                            "Attrition Rate: <b>%{z:.1%}</b><br>" +
                            "<extra></extra>"
            )
            
            st.plotly_chart(fig_region, use_container_width=True)
            
            # Add summary statistics
            st.markdown("""
                <div style='background-color: #f8fafc; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                    <h4>Region Summary</h4>
                    <p>Hover over the heatmap to see detailed attrition rates for each Region.</p>
                    <p>The color intensity indicates the attrition rate:</p>
                    <ul>
                        <li><span style='color: #e74c3c;'>Red</span> - High attrition (>30%)</li>
                        <li><span style='color: #f1c40f;'>Yellow</span> - Medium attrition (15-30%)</li>
                        <li><span style='color: #2ecc71;'>Green</span> - Low attrition (<15%)</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Region data is not available in the current dataset.")
    
    # Risk Factor Selection
    st.markdown('<h2 class="sub-header">Select Risk Factor</h2>', unsafe_allow_html=True)
    selected_factor = st.selectbox(
        "Choose a risk factor to analyze",
        options=features,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    # Risk Distribution by Selected Factor
    st.markdown('<h2 class="sub-header">Risk Distribution by {}</h2>'.format(selected_factor.replace('_', ' ').title()), unsafe_allow_html=True)
    
    # Create a DataFrame with the selected factor and risk scores
    factor_df = pd.DataFrame({
        'Factor': filtered_transformed_df[selected_factor],
        'Risk Score': filtered_probabilities * 100  # Convert to percentage
    })
    
    # Create a scatter plot without trendline
    fig = px.scatter(
        factor_df,
        x='Factor',
        y='Risk Score',
        title=f"Risk Scores vs {selected_factor.replace('_', ' ').title()}",
        labels={'Factor': selected_factor.replace('_', ' ').title(), 'Risk Score': 'Attrition Risk (%)'},
        color='Risk Score',
        color_continuous_scale=['#2ecc71', '#f1c40f', '#e74c3c']  # Green to Yellow to Red
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Poppins')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display filtered employees at risk based on the selected factor
    st.markdown('<h2 class="sub-header">Filtered Employees at Risk</h2>', unsafe_allow_html=True)
    
    # Create a DataFrame with employee details - use the same adjusted probabilities
    risk_df = pd.DataFrame({
        'Employee ID': filtered_df.index,
        'Risk Score': filtered_probabilities * 100,  # Convert to percentage
        selected_factor: filtered_transformed_df[selected_factor]
    }).sort_values('Risk Score', ascending=False)
    
    # Add clickable rows
    for idx, row in risk_df.iterrows():
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        with col1:
            st.write(f"Employee ID: {row['Employee ID']}")
        with col2:
            st.write(f"Risk Score: {row['Risk Score']:.0f}%")
        with col4:
            if st.button("View Details", key=f"view_{row['Employee ID']}"):
                st.session_state.selected_employee = row['Employee ID']
                st.session_state.active_tab = 1  # Set to Employee Analysis tab index
                st.experimental_rerun()

def display_employee_analysis_tab(df, transformed_df, probabilities, explanations, features, model):
    """Display the employee analysis tab with detailed insights"""
    # Add back button at the top
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back to Overview"):
            st.session_state.active_tab = 0
            st.experimental_rerun()
    
    st.markdown('<h1 class="main-header">Employee Analysis</h1>', unsafe_allow_html=True)
    
    # Employee Selection - automatically select the employee from session state
    st.markdown('<h2 class="sub-header">Select Employee</h2>', unsafe_allow_html=True)
    selected_employee = st.session_state.get('selected_employee', df.index[0])
    employee_id = st.selectbox(
        "Choose an employee to analyze",
        options=df.index,
        index=df.index.get_loc(selected_employee),
        format_func=lambda x: f"Employee ID: {x}"
    )
    
    # Get the index position of the selected employee
    employee_idx = df.index.get_loc(employee_id)
    
    # Employee Details Card
    employee_data = transformed_df.iloc[employee_idx]
    # Use raw probabilities without scaling
    risk_score = probabilities[employee_idx]
    
    # Get raw data for display
    raw_data = df.iloc[employee_idx]
    
    # Format risk level and color with consistent ranges
    if risk_score >= 0.70:  # High risk threshold
        risk_level = "High Risk"
        risk_color = "#dc2626"
    elif risk_score >= 0.50:  # Medium risk threshold
        risk_level = "Medium Risk"
        risk_color = "#f59e0b"
    else:  # Low risk
        risk_level = "Low Risk"
        risk_color = "#059669"
    
    # Display employee overview with raw data
    st.markdown(f"""
        <div class="stCard">
            <h3>Employee Overview</h3>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <p><strong>Employee ID:</strong> {employee_id}</p>
                    <p><strong>Tenure:</strong> {raw_data['Tenure']:.1f} years</p>
                    <p><strong>Performance Score:</strong> {raw_data['PastPerformance']:.1f}/2.0</p>
                    <p><strong>Training Hours:</strong> {raw_data['TrainingParticipation']:.1f} hours/year</p>
                    <p><strong>Engagement Score:</strong> {(raw_data['EngagementScore']*100):.1f}%</p>
                </div>
                <div style='text-align: right;'>
                    <h4>Attrition Risk</h4>
                    <div style='color: {risk_color}; font-weight: 600; padding: 0.5rem 1rem; border-radius: 0.5rem; background-color: {risk_color}20;'>
                        {risk_level} ({risk_score*100:.0f}%)
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Predictive Dashboard
    st.markdown('<h2 class="sub-header">Predictive Dashboard</h2>', unsafe_allow_html=True)
    
    # Create columns for different views
    col1, col2 = st.columns(2)
    
    # Get Business Unit information
    bu_columns = [col for col in df.columns if col.startswith('Business_Unit_')]
    employee_bu = next((col.replace('Business_Unit_', '') for col in bu_columns if raw_data[col] == 1), 'Unknown')
    
    # Calculate team and BU metrics
    with col1:
        # Team/Role Risk Overview
        role_columns = [col for col in df.columns if col.startswith('RoleHistory_')]
        employee_role = next((col.replace('RoleHistory_', '') for col in role_columns if raw_data[col] == 1), 'Unknown')
        
        role_risk = df[[col for col in df.columns if col.startswith('RoleHistory_')]].mean().mean()
        role_changes = max(0, float(raw_data['RoleChanges']))  # Ensure not negative
        peer_attrition = max(0, min(1.0, float(raw_data['TeamAttritionRate'])))  # Clamp between 0 and 1.0
        
        st.markdown(f"""
            <div class="stCard">
                <h4>Role Risk Overview</h4>
                <p><strong>Current Role:</strong> {employee_role}</p>
                <p><strong>Role Changes:</strong> {role_changes:.1f} per year</p>
                <p><strong>Role Attrition Rate:</strong> {role_risk:.1%}</p>
                <p><strong>Peer Attrition (60d):</strong> {peer_attrition:.1%}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Business Unit Risk
        bu_risk = df[df[f'Business_Unit_{employee_bu}'] == 1]['AttritionLabel'].mean() if employee_bu != 'Unknown' else 0
        bu_risk = max(0, min(1.0, float(bu_risk)))  # Clamp between 0 and 1.0
        st.markdown(f"""
            <div class="stCard">
                <h4>Business Unit Risk Overview</h4>
                <p><strong>Business Unit:</strong> {employee_bu}</p>
                <p><strong>BU Size:</strong> {len(df[df[f'Business_Unit_{employee_bu}'] == 1]) if employee_bu != 'Unknown' else 0}</p>
                <p><strong>BU Attrition Rate:</strong> {bu_risk:.1%}</p>
                <p><strong>High Risk Members:</strong> {len(df[(df[f'Business_Unit_{employee_bu}'] == 1) & (probabilities >= 0.55)]) if employee_bu != 'Unknown' else 0}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Confidence Level and Key Drivers
    st.markdown('<h2 class="sub-header">Confidence Level & Key Drivers</h2>', unsafe_allow_html=True)
    
    if explanations is not None:
        employee_explanation = explanations[employee_idx]
        
        # Calculate feature importance based on absolute SHAP values
        impact_df = pd.DataFrame({
            'Feature': features,
            'Impact': employee_explanation,
            'Absolute Impact': np.abs(employee_explanation)
        }).sort_values('Absolute Impact', ascending=False)
        
        # Calculate confidence level based on feature importance distribution
        confidence_score = 1 - (impact_df['Absolute Impact'].std() / impact_df['Absolute Impact'].mean())
        confidence_level = "High" if confidence_score > 0.7 else "Medium" if confidence_score > 0.4 else "Low"
        
        st.markdown(f"""
            <div class="stCard">
                <h4>Prediction Confidence</h4>
                <p><strong>Confidence Level:</strong> {confidence_level}</p>
                <p><strong>Confidence Score:</strong> {confidence_score:.2f}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Display key drivers
        st.markdown('<h4>Key Risk Drivers</h4>', unsafe_allow_html=True)
        for _, row in impact_df.head(5).iterrows():
            feature = row['Feature']
            impact = row['Impact']
            value = employee_data[feature]
            
            # Format value based on feature type
            if 'score' in feature and 'performance' in feature:
                formatted_value = f"{value:.1f}/5.0"
            elif 'rate' in feature or 'score' in feature:
                formatted_value = f"{value:.1%}"
            elif 'hours' in feature or 'years' in feature:
                formatted_value = f"{value:.1f}"
            else:
                formatted_value = f"{value:.2f}"
            
            # Set color based on impact
            if impact > 0:
                color = '#ffebee'  # Light red
                impact_text = "increases"
            else:
                color = '#e8f5e9'  # Light green
                impact_text = "decreases"
            
            st.markdown(f"""
                <div style='background-color: {color}; padding: 10px; margin: 5px 0; border-radius: 5px;'>
                    <p><strong>{feature.replace('_', ' ').title()}:</strong> {formatted_value}</p>
                    <p style='font-size: 0.9em; color: {'#dc2626' if impact > 0 else '#059669'};'>
                        {impact_text} attrition risk by {abs(impact):.3f}
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    # Risk Alerts
    st.markdown('<h2 class="sub-header">Risk Alerts</h2>', unsafe_allow_html=True)
    
    # Check for role-based risk cluster
    role_columns = [col for col in df.columns if col.startswith('RoleHistory_')]
    employee_role = next((col for col in role_columns if raw_data[col] == 1), None)
    
    if employee_role:
        role_high_risk = df[(df[employee_role] == 1) & (probabilities >= 0.55)]
        if len(role_high_risk) >= 3:
    # Check for role-based risk cluster
    role_columns = [col for col in df.columns if col.startswith('RoleHistory_')]
    employee_role = next((col for col in role_columns if raw_data[col] == 1), None)
    
    if employee_role:
        role_high_risk = df[(df[employee_role] == 1) & (probabilities >= 0.55)]
        if len(role_high_risk) >= 3:
            st.markdown(f"""
                <div style='background-color: #ffebee; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                    <h5>⚠️ High Risk Role Alert</h5>
                    <p><strong>Alert Level:</strong> Critical</p>
                    <p><strong>Details:</strong> {len(role_high_risk)} employees in the same role are at high risk of attrition.</p>
                    <p><strong>Recommended Action:</strong> Review role-specific challenges and career progression opportunities.</p>
                <div style='background-color: #ffebee; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                    <h5>⚠️ High Risk Role Alert</h5>
                    <p><strong>Alert Level:</strong> Critical</p>
                    <p><strong>Details:</strong> {len(role_high_risk)} employees in the same role are at high risk of attrition.</p>
                    <p><strong>Recommended Action:</strong> Review role-specific challenges and career progression opportunities.</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Check for BU risk cluster
    if employee_bu != 'Unknown':
        bu_high_risk = df[(df[f'Business_Unit_{employee_bu}'] == 1) & (probabilities >= 0.55)]
        if len(bu_high_risk) >= 5:
            st.markdown(f"""
                <div style='background-color: #ffebee; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                    <h5>⚠️ High Risk Business Unit Alert</h5>
                    <p><strong>Alert Level:</strong> Critical</p>
                    <p><strong>Details:</strong> {len(bu_high_risk)} members of {employee_bu} are at high risk of attrition.</p>
                    <p><strong>Recommended Action:</strong> Review BU leadership and organizational structure.</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Manager's Action Items
    st.markdown('<h2 class="sub-header">Manager\'s Action Items</h2>', unsafe_allow_html=True)
    
    # Get recommendations for this employee - use same thresholds as overview tab
    risk_level = "High Risk" if risk_score >= 0.70 else "Medium Risk" if risk_score >= 0.50 else "Low Risk"
    
    # Create a DataFrame with just this employee's data
    employee_df = transformed_df.loc[[employee_id]]
    employee_explanations = explanations[transformed_df.index == employee_id]
    
    # Generate recommendations
    recommendations = generate_recommendations(
        employee_df,
        employee_explanations,
        features,
        risk_level
    )
    
    if recommendations:
        for rec in recommendations:
            # Set color based on priority
            if rec['priority'] == 'Critical':
                color = '#ffebee'  # Light red
            elif rec['priority'] == 'High':
                color = '#fff3e0'  # Light orange
            else:
                color = '#e8f5e9'  # Light green
            
            st.markdown(f"""
                <div style='background-color: {color}; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                    <h5>{rec['action']}</h5>
                    <p><strong>Priority:</strong> {rec['priority']}</p>
                    <p><strong>Timeline:</strong> {rec['timeline']}</p>
                    <p>{rec['details']}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        # Add default recommendations based on risk level and key drivers
        if risk_level == "High Risk":
            st.markdown(f"""
                <div style='background-color: #ffebee; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                    <h5>Immediate Action Required</h5>
                    <p><strong>Priority:</strong> Critical</p>
                    <p><strong>Timeline:</strong> Within 24 hours</p>
                    <p><strong>Recommended Actions:</strong></p>
                    <ul>
                        <li>Schedule an immediate one-on-one meeting</li>
                        <li>Review performance and career progression</li>
                        <li>Address engagement concerns</li>
                        <li>Develop retention action plan</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        elif risk_level == "Medium Risk":
            st.markdown(f"""
                <div style='background-color: #fff3e0; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                    <h5>Proactive Intervention Needed</h5>
                    <p><strong>Priority:</strong> High</p>
                    <p><strong>Timeline:</strong> Within 1 week</p>
                    <p><strong>Recommended Actions:</strong></p>
                    <ul>
                        <li>Schedule regular check-ins</li>
                        <li>Review engagement factors</li>
                        <li>Discuss career development opportunities</li>
                        <li>Monitor workload and satisfaction</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

def transform_data(df):
    """Transform the data to match the model's expected features"""
    # Convert EmployeeID to string format
    df.index = df.index.astype(str)
    
    # Define the exact feature names expected by the model (23 features)
    expected_features = [
        'tenure', 'performance_score', 'last_promotion', 'training_hours', 'role_changes',
        'survey_satisfaction', 'peer_attrition_last60d', 'leave_days_taken', 'skill_relevance',
        'team_attrition_rate', 'dept_attrition_rate', 'manager_avg_performance', 'manager_team_size',
        'promotion_delay', 'never_promoted', 'engagement_score', 'stagnation_risk',
        'performance_above_dept_avg', 'leave_utilization', 'high_performer_no_promo',
        'Business_Unit_BU1', 'Business_Unit_BU2', 'Business_Unit_BU3'
    ]
    
    # Create transformed features DataFrame with default values
    transformed_df = pd.DataFrame(index=df.index)
    for feature in expected_features:
        transformed_df[feature] = 0.0
    
    # Map the input data to the expected features
    feature_mappings = {
        'Tenure': 'tenure',
        'PastPerformance': 'performance_score',
        'LastPromotionYearsAgo': 'last_promotion',
        'TrainingParticipation': 'training_hours',
        'RoleChanges': 'role_changes',
        'FeedbackScore': 'survey_satisfaction',
        'TeamAttritionRate': 'peer_attrition_last60d',
        'LeavePattern': 'leave_days_taken',
        'SkillRelevance': 'skill_relevance',
        'TeamAttritionRate': 'team_attrition_rate',
        'TeamAttritionRate': 'dept_attrition_rate',  # Using team rate as proxy for dept rate
        'Feedback360': 'manager_avg_performance',
        'TeamSize': 'manager_team_size',
        'LastPromotionYearsAgo': 'promotion_delay',
        'Promotions': 'never_promoted',
        'EngagementScore': 'engagement_score',
        'LastPromotionYearsAgo': 'stagnation_risk',
        'PastPerformance': 'performance_above_dept_avg',
        'LeavePattern': 'leave_utilization',
        'Promotions': 'high_performer_no_promo'
    }
    
    # Apply the mappings
    for original_col, new_col in feature_mappings.items():
        if original_col in df.columns:
            transformed_df[new_col] = df[original_col]
    
    # Handle derived features
    transformed_df['stagnation_risk'] = (transformed_df['last_promotion'] > 3).astype(float)
    transformed_df['performance_above_dept_avg'] = (transformed_df['performance_score'] > transformed_df['performance_score'].mean()).astype(float)
    transformed_df['high_performer_no_promo'] = ((transformed_df['performance_score'] > 4) & (transformed_df['never_promoted'] == 1)).astype(float)
    
    # Handle Business Unit columns
    bu_columns = [col for col in df.columns if col.startswith('Business_Unit_')]
    if bu_columns:
        for bu_col in bu_columns[:3]:  # Only use first 3 Business Unit columns
            if bu_col in expected_features:
                transformed_df[bu_col] = df[bu_col]
    
    # Check for BU risk cluster
    if employee_bu != 'Unknown':
        bu_high_risk = df[(df[f'Business_Unit_{employee_bu}'] == 1) & (probabilities >= 0.55)]
        if len(bu_high_risk) >= 5:
            st.markdown(f"""
                <div style='background-color: #ffebee; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                    <h5>⚠️ High Risk Business Unit Alert</h5>
                    <p><strong>Alert Level:</strong> Critical</p>
                    <p><strong>Details:</strong> {len(bu_high_risk)} members of {employee_bu} are at high risk of attrition.</p>
                    <p><strong>Recommended Action:</strong> Review BU leadership and organizational structure.</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Manager's Action Items
    st.markdown('<h2 class="sub-header">Manager\'s Action Items</h2>', unsafe_allow_html=True)
    
    # Get recommendations for this employee - use same thresholds as overview tab
    risk_level = "High Risk" if risk_score >= 0.70 else "Medium Risk" if risk_score >= 0.50 else "Low Risk"
    
    # Create a DataFrame with just this employee's data
    employee_df = transformed_df.loc[[employee_id]]
    employee_explanations = explanations[transformed_df.index == employee_id]
    
    # Generate recommendations
    recommendations = generate_recommendations(
        employee_df,
        employee_explanations,
        features,
        risk_level
    )
    
    if recommendations:
        for rec in recommendations:
            # Set color based on priority
            if rec['priority'] == 'Critical':
                color = '#ffebee'  # Light red
            elif rec['priority'] == 'High':
                color = '#fff3e0'  # Light orange
            else:
                color = '#e8f5e9'  # Light green
            
            st.markdown(f"""
                <div style='background-color: {color}; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                    <h5>{rec['action']}</h5>
                    <p><strong>Priority:</strong> {rec['priority']}</p>
                    <p><strong>Timeline:</strong> {rec['timeline']}</p>
                    <p>{rec['details']}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        # Add default recommendations based on risk level and key drivers
        if risk_level == "High Risk":
            st.markdown(f"""
                <div style='background-color: #ffebee; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                    <h5>Immediate Action Required</h5>
                    <p><strong>Priority:</strong> Critical</p>
                    <p><strong>Timeline:</strong> Within 24 hours</p>
                    <p><strong>Recommended Actions:</strong></p>
                    <ul>
                        <li>Schedule an immediate one-on-one meeting</li>
                        <li>Review performance and career progression</li>
                        <li>Address engagement concerns</li>
                        <li>Develop retention action plan</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        elif risk_level == "Medium Risk":
            st.markdown(f"""
                <div style='background-color: #fff3e0; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                    <h5>Proactive Intervention Needed</h5>
                    <p><strong>Priority:</strong> High</p>
                    <p><strong>Timeline:</strong> Within 1 week</p>
                    <p><strong>Recommended Actions:</strong></p>
                    <ul>
                        <li>Schedule regular check-ins</li>
                        <li>Review engagement factors</li>
                        <li>Discuss career development opportunities</li>
                        <li>Monitor workload and satisfaction</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

def transform_data(df):
    """Transform the data to match the model's expected features"""
    # Convert EmployeeID to string format
    df.index = df.index.astype(str)
    
    # Define the exact feature names expected by the model (23 features)
    expected_features = [
        'tenure', 'performance_score', 'last_promotion', 'training_hours', 'role_changes',
        'survey_satisfaction', 'peer_attrition_last60d', 'leave_days_taken', 'skill_relevance',
        'team_attrition_rate', 'dept_attrition_rate', 'manager_avg_performance', 'manager_team_size',
        'promotion_delay', 'never_promoted', 'engagement_score', 'stagnation_risk',
        'performance_above_dept_avg', 'leave_utilization', 'high_performer_no_promo',
        'Business_Unit_BU1', 'Business_Unit_BU2', 'Business_Unit_BU3'
    ]
    
    # Create transformed features DataFrame with default values
    transformed_df = pd.DataFrame(index=df.index)
    for feature in expected_features:
        transformed_df[feature] = 0.0
    
    # Map the input data to the expected features
    feature_mappings = {
        'Tenure': 'tenure',
        'PastPerformance': 'performance_score',
        'LastPromotionYearsAgo': 'last_promotion',
        'TrainingParticipation': 'training_hours',
        'RoleChanges': 'role_changes',
        'FeedbackScore': 'survey_satisfaction',
        'TeamAttritionRate': 'peer_attrition_last60d',
        'LeavePattern': 'leave_days_taken',
        'SkillRelevance': 'skill_relevance',
        'TeamAttritionRate': 'team_attrition_rate',
        'TeamAttritionRate': 'dept_attrition_rate',  # Using team rate as proxy for dept rate
        'Feedback360': 'manager_avg_performance',
        'TeamSize': 'manager_team_size',
        'LastPromotionYearsAgo': 'promotion_delay',
        'Promotions': 'never_promoted',
        'EngagementScore': 'engagement_score',
        'LastPromotionYearsAgo': 'stagnation_risk',
        'PastPerformance': 'performance_above_dept_avg',
        'LeavePattern': 'leave_utilization',
        'Promotions': 'high_performer_no_promo'
    }
    
    # Apply the mappings
    for original_col, new_col in feature_mappings.items():
        if original_col in df.columns:
            transformed_df[new_col] = df[original_col]
    
    # Handle derived features
    transformed_df['stagnation_risk'] = (transformed_df['last_promotion'] > 3).astype(float)
    transformed_df['performance_above_dept_avg'] = (transformed_df['performance_score'] > transformed_df['performance_score'].mean()).astype(float)
    transformed_df['high_performer_no_promo'] = ((transformed_df['performance_score'] > 4) & (transformed_df['never_promoted'] == 1)).astype(float)
    
    # Handle Business Unit columns
    bu_columns = [col for col in df.columns if col.startswith('Business_Unit_')]
    if bu_columns:
        for bu_col in bu_columns[:3]:  # Only use first 3 Business Unit columns
            if bu_col in expected_features:
                transformed_df[bu_col] = df[bu_col]
    else:
        # If no Business Unit columns exist, create dummy columns for first 3
        for i in range(1, 4):
            col_name = f'Business_Unit_BU{i}'
            if col_name in expected_features:
                transformed_df[col_name] = 0.0
    
    # Ensure all expected features are present and in the correct order
    transformed_df = transformed_df[expected_features]
    
    # Add small random noise to constant columns to prevent validation errors
    for col in transformed_df.columns:
        if transformed_df[col].nunique() == 1:
            noise = np.random.normal(0, 0.0001, size=len(transformed_df))
            transformed_df[col] = transformed_df[col] + noise
    
    return transformed_df

def main():
    """Main function to run the Streamlit app"""
    # Initialize session state for tab selection and risk threshold
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0  # 0 for Overview, 1 for Employee Analysis
    if 'risk_threshold' not in st.session_state:
        st.session_state.risk_threshold = 40
    if 'selected_employee' not in st.session_state:
        st.session_state.selected_employee = None
    
    # Load data and resources
    df = load_data()
    if df is None:
        st.error("Failed to load data. Please check the data file and try again.")
        return
    
    # Set the index to EmployeeID column
    df.set_index('EmployeeID', inplace=True)
    
    # Transform data
    transformed_df = transform_data(df)
    
    # Define features list based on transformed columns
    features = transformed_df.columns.tolist()
    
    # Validate data
    is_valid, validation_message = validate_data(transformed_df, features)
    if not is_valid:
        st.error(f"Data validation failed: {validation_message}")
        return
    
    # Preprocess data
    try:
        transformed_df = preprocess_data(transformed_df, features)
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return
    
    # Load model and explainer
    model, explainer = load_prediction_resources()
    if model is None or explainer is None:
        st.error("Failed to load prediction resources. Please check the model files and try again.")
        return
    
    # Generate predictions
    try:
        probabilities, explanations = predict_all_employees(transformed_df, model, explainer, features)
        if probabilities is None or explanations is None:
            st.error("Failed to generate predictions. Please check the model and data.")
            return
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
        return
    
    # Create tabs
    tab1, tab2 = st.tabs(["Overview", "Employee Analysis"])
    
    # Display content based on active tab
    if st.session_state.active_tab == 0:
        with tab1:
            display_overview_tab(df, transformed_df, probabilities, features, explanations)
    else:
        with tab2:
            display_employee_analysis_tab(df, transformed_df, probabilities, explanations, features, model)
        # If no Business Unit columns exist, create dummy columns for first 3
        for i in range(1, 4):
            col_name = f'Business_Unit_BU{i}'
            if col_name in expected_features:
                transformed_df[col_name] = 0.0
    
    # Ensure all expected features are present and in the correct order
    transformed_df = transformed_df[expected_features]
    
    # Add small random noise to constant columns to prevent validation errors
    for col in transformed_df.columns:
        if transformed_df[col].nunique() == 1:
            noise = np.random.normal(0, 0.0001, size=len(transformed_df))
            transformed_df[col] = transformed_df[col] + noise
    
    return transformed_df

def main():
    """Main function to run the Streamlit app"""
    # Initialize session state for tab selection and risk threshold
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0  # 0 for Overview, 1 for Employee Analysis
    if 'risk_threshold' not in st.session_state:
        st.session_state.risk_threshold = 40
    if 'selected_employee' not in st.session_state:
        st.session_state.selected_employee = None
    
    # Load data and resources
    df = load_data()
    if df is None:
        st.error("Failed to load data. Please check the data file and try again.")
        return
    
    # Set the index to EmployeeID column
    df.set_index('EmployeeID', inplace=True)
    
    # Transform data
    transformed_df = transform_data(df)
    
    # Define features list based on transformed columns
    features = transformed_df.columns.tolist()
    
    # Validate data
    is_valid, validation_message = validate_data(transformed_df, features)
    if not is_valid:
        st.error(f"Data validation failed: {validation_message}")
        return
    
    # Preprocess data
    try:
        transformed_df = preprocess_data(transformed_df, features)
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return
    
    # Load model and explainer
    model, explainer = load_prediction_resources()
    if model is None or explainer is None:
        st.error("Failed to load prediction resources. Please check the model files and try again.")
        return
    
    # Generate predictions
    try:
        probabilities, explanations = predict_all_employees(transformed_df, model, explainer, features)
        if probabilities is None or explanations is None:
            st.error("Failed to generate predictions. Please check the model and data.")
            return
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
        return
    
    # Create tabs
    tab1, tab2 = st.tabs(["Overview", "Employee Analysis"])
    
    # Display content based on active tab
    if st.session_state.active_tab == 0:
        with tab1:
            display_overview_tab(df, transformed_df, probabilities, features, explanations)
    else:
        with tab2:
            display_employee_analysis_tab(df, transformed_df, probabilities, explanations, features, model)

if __name__ == "__main__":
    main()