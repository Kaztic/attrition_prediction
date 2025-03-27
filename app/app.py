#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st

# Set page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Custom CSS with modern design
st.markdown("""
<style>
    /* Modern Typography */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Main Layout */
    .main {
        background-color: #f8fafc;
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    /* Headers */
    .main-header {
        font-family: 'Poppins', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-family: 'Poppins', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
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
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        transition: transform 0.2s;
        max-width: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-family: 'Poppins', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
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
    }
    
    .risk-medium {
        color: #f59e0b;
        font-weight: 600;
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
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        color: #64748b;
        padding: 0.75rem 1.5rem;
        border-radius: 0.75rem;
        transition: all 0.2s;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
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
    }
    
    /* Data Tables */
    .dataframe {
        font-family: 'Poppins', sans-serif;
        border-radius: 1rem;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        background-color: white;
        max-width: 100%;
    }
    
    .dataframe thead th {
        background-color: #f8fafc;
        font-weight: 600;
        color: #1e293b;
        padding: 1rem;
    }
    
    .dataframe tbody td {
        padding: 1rem;
        color: #64748b;
        background-color: white;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: #f8fafc;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: white;
        border-right: 1px solid #e2e8f0;
    }
    
    .sidebar .sidebar-content {
        padding: 2rem;
    }
    
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
        
        .main-header {
            font-size: 2rem;
        }
        
        .sub-header {
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

@st.cache_data
def load_data():
    """Load and cache the preprocessed employee data"""
    try:
        df = pd.read_csv('data/preprocessed_employee_data.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_prediction_resources():
    """Load and cache the model and explainer"""
    try:
        model = AttritionPredictionModel.load_model()
        explainer = shap.TreeExplainer(model)
        return model, explainer
    except Exception as e:
        st.error(f"Error loading prediction resources: {str(e)}")
        return None, None

@st.cache_data
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

@st.cache_data
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

@st.cache_data
def predict_all_employees(df, _model, _explainer, features):
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
            return None, None, None
        
        # Use only the required features for prediction
        X = df[required_features]
        
        # Generate predictions
        probabilities = _model.predict_proba(X)[:, 1]
        
        # Apply consistent risk score adjustment (multiply by 1.5, cap at 1.0)
        adjusted_probabilities = np.minimum(probabilities * 1.5, np.ones_like(probabilities))
        
        # Generate SHAP explanations
        explanations = _explainer.shap_values(X)
        if isinstance(explanations, list):
            explanations = explanations[0]  # For binary classification, get first class
            
        # Create a full explanations array with zeros for categorical features
        full_explanations = np.zeros((len(df), len(features)))
        for i, feature in enumerate(features):
            if feature in required_features:
                feature_idx = required_features.index(feature)
                full_explanations[:, i] = explanations[:, feature_idx]
        
        return probabilities, adjusted_probabilities, full_explanations
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
        return None, None, None

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

def display_overview_tab(df, transformed_df, probabilities, adjusted_probabilities, features, explanations):
    """Display the overview tab with key metrics and visualizations"""
    st.markdown('<h1 class="main-header">Employee Attrition Analytics</h1>', unsafe_allow_html=True)
    
    # Employee Search
    with st.container():
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
    
    # Calculate metrics once (these are small operations)
    total_count = len(df)
    high_risk_count = (adjusted_probabilities >= 0.70).sum()
    medium_risk_count = ((adjusted_probabilities >= 0.50) & (adjusted_probabilities < 0.70)).sum()
    low_risk_count = (adjusted_probabilities < 0.50).sum()
    
    # Key Metrics with clickable cards
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("All Employees", key="all_btn"):
                st.session_state.risk_filter = 'all'
                # Clear heatmap caches
                for key in list(st.session_state.keys()):
                    if key.startswith('bu_heatmap_') or key.startswith('region_heatmap_'):
                        del st.session_state[key]
                st.experimental_rerun()
            st.markdown(f"""
                <div class="metric-card" style="cursor: pointer;">
                    <div class="metric-value">{total_count}</div>
                    <div class="metric-label">All Employees</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("High Risk", key="high_risk_btn"):
                st.session_state.risk_filter = 'high'
                # Clear heatmap caches
                for key in list(st.session_state.keys()):
                    if key.startswith('bu_heatmap_') or key.startswith('region_heatmap_'):
                        del st.session_state[key]
                st.experimental_rerun()
            st.markdown(f"""
                <div class="metric-card" style="cursor: pointer;">
                    <div class="metric-value">{high_risk_count}</div>
                    <div class="metric-label">High Risk Employees</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if st.button("Medium Risk", key="medium_risk_btn"):
                st.session_state.risk_filter = 'medium'
                # Clear heatmap caches
                for key in list(st.session_state.keys()):
                    if key.startswith('bu_heatmap_') or key.startswith('region_heatmap_'):
                        del st.session_state[key]
                st.experimental_rerun()
            st.markdown(f"""
                <div class="metric-card" style="cursor: pointer;">
                    <div class="metric-value">{medium_risk_count}</div>
                    <div class="metric-label">Medium Risk Employees</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            if st.button("Low Risk", key="low_risk_btn"):
                st.session_state.risk_filter = 'low'
                # Clear heatmap caches
                for key in list(st.session_state.keys()):
                    if key.startswith('bu_heatmap_') or key.startswith('region_heatmap_'):
                        del st.session_state[key]
                st.experimental_rerun()
            st.markdown(f"""
                <div class="metric-card" style="cursor: pointer;">
                    <div class="metric-value">{low_risk_count}</div>
                    <div class="metric-label">Low Risk Employees</div>
                </div>
            """, unsafe_allow_html=True)
    
    # Filter data based on selected risk level - do this only once
    if st.session_state.risk_filter == 'high':
        filtered_mask = adjusted_probabilities >= 0.70
    elif st.session_state.risk_filter == 'medium':
        filtered_mask = (adjusted_probabilities >= 0.50) & (adjusted_probabilities < 0.70)
    elif st.session_state.risk_filter == 'low':
        filtered_mask = adjusted_probabilities < 0.50
    else:  # 'all'
        filtered_mask = np.ones(len(df), dtype=bool)
    
    # Apply filter only once and store results
    if 'filtered_data' not in st.session_state or st.session_state.get('last_filter') != st.session_state.risk_filter:
        filtered_df = df[filtered_mask]
        filtered_probabilities = probabilities[filtered_mask]
        filtered_adjusted_probabilities = adjusted_probabilities[filtered_mask]
        filtered_transformed_df = transformed_df[filtered_mask]
        
        # Store in session state to avoid recalculation
        st.session_state.filtered_data = {
            'df': filtered_df,
            'probabilities': filtered_probabilities,
            'adjusted_probabilities': filtered_adjusted_probabilities,
            'transformed_df': filtered_transformed_df
        }
        st.session_state.last_filter = st.session_state.risk_filter
    else:
        # Retrieve from session state
        filtered_df = st.session_state.filtered_data['df']
        filtered_probabilities = st.session_state.filtered_data['probabilities']
        filtered_adjusted_probabilities = st.session_state.filtered_data['adjusted_probabilities']
        filtered_transformed_df = st.session_state.filtered_data['transformed_df']
    
    # If the filtered data is large, limit display for performance
    MAX_DISPLAY = 100
    display_limit = min(len(filtered_df), MAX_DISPLAY)
    
    # Business Unit and Region Heatmaps - only show if not too many in filtered set
    if len(filtered_df) < 1000 or st.session_state.risk_filter == 'all':  # Only for manageable datasets
        with st.container():
            st.markdown('<h2 class="sub-header">Attrition Heatmaps</h2>', unsafe_allow_html=True)
            
            # Add debug information in expander
            with st.expander("Debug Info", expanded=False):
                st.write(f"Current risk filter: {st.session_state.risk_filter}")
                st.write(f"Filtered data size: {len(filtered_df)}")
                st.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}")
            
            # Create tabs for different heatmap views
            heatmap_tab1, heatmap_tab2 = st.tabs(["Business Unit Heatmap", "Region Heatmap"])
            
            with heatmap_tab1:
                # Use a button to force refresh if needed
                if st.button("Refresh BU Heatmap", key="refresh_bu_heatmap"):
                    # Clear the cache for this specific visualization
                    if "bu_heatmap_data" in st.session_state:
                        del st.session_state["bu_heatmap_data"]
                
                # Business Unit Heatmap
                # Get all Business Unit columns
                bu_columns = [col for col in df.columns if col.startswith('Business_Unit_')]
                if bu_columns:
                    # Add option to show all BUs
                    show_all_bus = st.checkbox("Show all Business Units", value=False, key="show_all_bus")
                    
                    # Calculate or retrieve heatmap data
                    bu_heatmap_key = f"bu_heatmap_{st.session_state.risk_filter}_{show_all_bus}"
                    if bu_heatmap_key not in st.session_state:
                        if show_all_bus:
                            # Show all business units
                            top_bu_columns = bu_columns
                        else:
                            # Show only top 5 by employee count
                            top_bu_columns = sorted(bu_columns, key=lambda x: df[x].sum(), reverse=True)[:5]
                            
                        bu_attrition = pd.DataFrame()
                        
                        for bu_col in top_bu_columns:
                            bu_name = bu_col.replace('Business_Unit_', '')
                            # Filter for employees in this BU
                            bu_employees = df[df[bu_col] == 1]
                            if not bu_employees.empty:
                                # Calculate overall attrition rate for this BU
                                bu_attrition[bu_name] = [bu_employees['AttritionLabel'].mean()]
                        
                        st.session_state[bu_heatmap_key] = bu_attrition
                    else:
                        bu_attrition = st.session_state[bu_heatmap_key]
                    
                    # Create heatmap with improved colors - make sure each chart has a unique key
                    if not bu_attrition.empty:
                        # Use timestamp in key to ensure uniqueness
                        chart_key = f"bu_chart_{datetime.now().timestamp()}"
                        
                        # Add option to sort by attrition rate
                        sort_by_rate = st.checkbox("Sort by attrition rate (highest first)", value=True, key="sort_bu_by_rate")
                        
                        if sort_by_rate:
                            # Sort columns by attrition rate (highest first)
                            sorted_columns = bu_attrition.loc[0].sort_values(ascending=False).index.tolist()
                            bu_attrition = bu_attrition[sorted_columns]
                        
                        # Add horizontal bar chart option
                        chart_type = st.radio("Chart type", ["Heatmap", "Bar chart"], key="bu_chart_type")
                        
                        if chart_type == "Heatmap":
                            fig_bu = px.imshow(
                                bu_attrition,
                                title="Attrition Rate by Business Unit",
                                labels=dict(x="Business Unit", y="", color="Attrition Rate"),
                                aspect="auto",
                                color_continuous_scale=['#2ecc71', '#f1c40f', '#e74c3c']  # Green to Yellow to Red
                            )
                            
                            # Update layout
                            fig_bu.update_layout(
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                font=dict(family='Poppins'),
                                xaxis_title="Business Unit",
                                yaxis_title="",
                                yaxis_visible=False,
                                coloraxis_colorbar_title="Attrition Rate",
                                height=300  # Reduced height for faster rendering
                            )
                            
                            # Add hover template
                            fig_bu.update_traces(
                                hovertemplate="<b>%{x}</b><br>" +
                                            "Attrition Rate: <b>%{z:.1%}</b><br>" +
                                            "<extra></extra>"
                            )
                        else:
                            # Create bar chart
                            bar_data = pd.DataFrame({
                                'Business Unit': bu_attrition.columns,
                                'Attrition Rate': bu_attrition.iloc[0].values
                            })
                            
                            fig_bu = px.bar(
                                bar_data,
                                x='Business Unit',
                                y='Attrition Rate',
                                title="Attrition Rate by Business Unit",
                                color='Attrition Rate',
                                color_continuous_scale=['#2ecc71', '#f1c40f', '#e74c3c']  # Green to Yellow to Red
                            )
                            
                            # Update layout
                            fig_bu.update_layout(
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                font=dict(family='Poppins'),
                                xaxis_title="Business Unit",
                                yaxis_title="Attrition Rate",
                                yaxis_tickformat='.1%',
                                height=400  # Slightly taller for bar chart
                            )
                            
                            # Add hover template
                            fig_bu.update_traces(
                                hovertemplate="<b>%{x}</b><br>" +
                                            "Attrition Rate: <b>%{y:.1%}</b><br>" +
                                            "<extra></extra>"
                            )
                        
                        st.plotly_chart(fig_bu, use_container_width=True, key=chart_key)
                        
                        # Show raw data in expandable section
                        with st.expander("View raw data"):
                            st.dataframe(bu_attrition)
                            
                        # Add option to show employee counts
                        if st.button("Show Employee Counts by Business Unit", key="show_bu_counts"):
                            # Create dataframe with BU counts
                            bu_counts = pd.DataFrame(index=["Count"])
                            for bu_col in top_bu_columns:
                                bu_name = bu_col.replace('Business_Unit_', '')
                                bu_counts[bu_name] = [df[bu_col].sum()]
                            
                            # Sort by count if desired
                            if sort_by_rate:
                                bu_counts = bu_counts[sorted_columns]
                            
                            # Show counts
                            st.subheader("Employee Counts by Business Unit")
                            st.dataframe(bu_counts)
                            
                            # Create bar chart of counts
                            count_data = pd.DataFrame({
                                'Business Unit': bu_counts.columns,
                                'Employee Count': bu_counts.iloc[0].values
                            })
                            
                            fig_count = px.bar(
                                count_data,
                                x='Business Unit',
                                y='Employee Count',
                                title="Employee Count by Business Unit",
                                color='Employee Count',
                                color_continuous_scale=['#d1d5db', '#6366f1']  # Grey to Purple
                            )
                            
                            # Update layout
                            fig_count.update_layout(
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                font=dict(family='Poppins'),
                                height=300
                            )
                            
                            st.plotly_chart(fig_count, use_container_width=True)
                        else:
                            st.info("Business Unit data is not available in the current dataset.")
            
            with heatmap_tab2:
                # Use a button to force refresh if needed
                if st.button("Refresh Region Heatmap", key="refresh_region_heatmap"):
                    # Clear the cache for this specific visualization
                    if "region_heatmap_data" in st.session_state:
                        del st.session_state["region_heatmap_data"]
                    
                # Region Heatmap - only calculate if this tab is selected
                # Get all Region columns
                region_columns = [col for col in df.columns if col.startswith('Region_')]
                if region_columns:
                    # Add option to show all regions
                    show_all_regions = st.checkbox("Show all Regions", value=False, key="show_all_regions")
                    
                    # Calculate or retrieve heatmap data
                    region_heatmap_key = f"region_heatmap_{st.session_state.risk_filter}_{show_all_regions}"
                    if region_heatmap_key not in st.session_state:
                        # Only display top regions by employee count for performance, or all if selected
                        if show_all_regions:
                            # Show all regions
                            top_region_columns = region_columns
                        else:
                            # Show only top 5 by employee count
                            top_region_columns = sorted(region_columns, key=lambda x: df[x].sum(), reverse=True)[:5]
                            
                        region_attrition = pd.DataFrame()
                        
                        for region_col in top_region_columns:
                            region_name = region_col.replace('Region_', '')
                            # Filter for employees in this Region
                            region_employees = df[df[region_col] == 1]
                            if not region_employees.empty:
                                # Calculate overall attrition rate for this Region
                                region_attrition[region_name] = [region_employees['AttritionLabel'].mean()]
                        
                        st.session_state[region_heatmap_key] = region_attrition
                    else:
                        region_attrition = st.session_state[region_heatmap_key]
                    
                    # Create heatmap with improved colors - make sure each chart has a unique key
                    if not region_attrition.empty:
                        # Use timestamp in key to ensure uniqueness
                        chart_key = f"region_chart_{datetime.now().timestamp()}"
                        
                        # Add option to sort by attrition rate
                        sort_by_rate = st.checkbox("Sort by attrition rate (highest first)", value=True, key="sort_region_by_rate")
                        
                        if sort_by_rate:
                            # Sort columns by attrition rate (highest first)
                            sorted_columns = region_attrition.loc[0].sort_values(ascending=False).index.tolist()
                            region_attrition = region_attrition[sorted_columns]
                        
                        # Add horizontal bar chart option
                        chart_type = st.radio("Chart type", ["Heatmap", "Bar chart"], key="region_chart_type")
                        
                        if chart_type == "Heatmap":
                            fig_region = px.imshow(
                                region_attrition,
                                title="Attrition Rate by Region",
                                labels=dict(x="Region", y="", color="Attrition Rate"),
                                aspect="auto",
                                color_continuous_scale=['#2ecc71', '#f1c40f', '#e74c3c']  # Green to Yellow to Red
                            )
                            
                            # Update layout
                            fig_region.update_layout(
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                font=dict(family='Poppins'),
                                xaxis_title="Region",
                                yaxis_title="",
                                yaxis_visible=False,
                                coloraxis_colorbar_title="Attrition Rate",
                                height=300  # Reduced height for faster rendering
                            )
                            
                            # Add hover template
                            fig_region.update_traces(
                                hovertemplate="<b>%{x}</b><br>" +
                                            "Attrition Rate: <b>%{z:.1%}</b><br>" +
                                            "<extra></extra>"
                            )
                        else:
                            # Create bar chart
                            bar_data = pd.DataFrame({
                                'Region': region_attrition.columns,
                                'Attrition Rate': region_attrition.iloc[0].values
                            })
                            
                            fig_region = px.bar(
                                bar_data,
                                x='Region',
                                y='Attrition Rate',
                                title="Attrition Rate by Region",
                                color='Attrition Rate',
                                color_continuous_scale=['#2ecc71', '#f1c40f', '#e74c3c']  # Green to Yellow to Red
                            )
                            
                            # Update layout
                            fig_region.update_layout(
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                font=dict(family='Poppins'),
                                xaxis_title="Region",
                                yaxis_title="Attrition Rate",
                                yaxis_tickformat='.1%',
                                height=400  # Slightly taller for bar chart
                            )
                            
                            # Add hover template
                            fig_region.update_traces(
                                hovertemplate="<b>%{x}</b><br>" +
                                            "Attrition Rate: <b>%{y:.1%}</b><br>" +
                                            "<extra></extra>"
                            )
                        
                        st.plotly_chart(fig_region, use_container_width=True, key=chart_key)
                        
                        # Show raw data in expandable section
                        with st.expander("View raw data"):
                            st.dataframe(region_attrition)
                            
                        # Add option to show employee counts
                        if st.button("Show Employee Counts by Region", key="show_region_counts"):
                            # Create dataframe with region counts
                            region_counts = pd.DataFrame(index=["Count"])
                            for region_col in top_region_columns:
                                region_name = region_col.replace('Region_', '')
                                region_counts[region_name] = [df[region_col].sum()]
                            
                            # Sort by count if desired
                            if sort_by_rate:
                                region_counts = region_counts[sorted_columns]
                            
                            # Show counts
                            st.subheader("Employee Counts by Region")
                            st.dataframe(region_counts)
                            
                            # Create bar chart of counts
                            count_data = pd.DataFrame({
                                'Region': region_counts.columns,
                                'Employee Count': region_counts.iloc[0].values
                            })
                            
                            fig_count = px.bar(
                                count_data,
                                x='Region',
                                y='Employee Count',
                                title="Employee Count by Region",
                                color='Employee Count',
                                color_continuous_scale=['#d1d5db', '#6366f1']  # Grey to Purple
                            )
                            
                            # Update layout
                            fig_count.update_layout(
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                font=dict(family='Poppins'),
                                height=300
                            )
                            
                            st.plotly_chart(fig_count, use_container_width=True)
                        else:
                            st.info("Region data is not available in the current dataset.")
            
    # Risk Factor Selection
    with st.container():
        st.markdown('<h2 class="sub-header">Select Risk Factor</h2>', unsafe_allow_html=True)
        # Use a smaller subset of features for better performance if dataset is large
        if len(filtered_df) > 500:
            # Use only the most important features for large datasets
            important_features = ['tenure', 'performance_score', 'engagement_score', 
                                'team_attrition_rate', 'last_promotion']
            display_features = [f for f in important_features if f in features]
            if not display_features:  # Fallback if important features aren't found
                display_features = features[:5]
        else:
            display_features = features
        
    selected_factor = st.selectbox(
        "Choose a risk factor to analyze",
        options=display_features,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    # Risk Distribution by Selected Factor
    st.markdown('<h2 class="sub-header">Risk Distribution by {}</h2>'.format(selected_factor.replace('_', ' ').title()), unsafe_allow_html=True)
    
    # Create a DataFrame with the selected factor and risk scores - limit to sample for performance
    if len(filtered_transformed_df) > 200:
        # Sample data for plotting to improve performance
        sample_indices = np.random.choice(len(filtered_transformed_df), 200, replace=False)
        plot_factor_df = pd.DataFrame({
            'Factor': filtered_transformed_df.iloc[sample_indices][selected_factor],
            'Risk Score': filtered_adjusted_probabilities[sample_indices] * 100  # Convert to percentage
        })
    else:
        plot_factor_df = pd.DataFrame({
            'Factor': filtered_transformed_df[selected_factor],
            'Risk Score': filtered_adjusted_probabilities * 100  # Convert to percentage
        })
    
    # Create a scatter plot without trendline
    fig = px.scatter(
        plot_factor_df,
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
        font=dict(family='Poppins'),
        height=400  # Reduced height for faster rendering
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display filtered employees at risk - only show top ones for performance
    with st.container():
        st.markdown('<h2 class="sub-header">Filtered Employees at Risk</h2>', unsafe_allow_html=True)
        
        # Create a DataFrame with employee details - use the adjusted probabilities
        risk_df = pd.DataFrame({
            'Employee ID': filtered_df.index,
            'Risk Score': filtered_adjusted_probabilities * 100,  # Convert to percentage
            selected_factor: filtered_transformed_df[selected_factor]
        }).sort_values('Risk Score', ascending=False)
        
        # Only display a limited number of employees for performance
        if display_limit < len(risk_df):
            st.info(f"Showing top {display_limit} of {len(risk_df)} employees. Filter to see more specific results.")
        
        # Add clickable rows - only for the top employees
        for idx, row in risk_df.head(display_limit).iterrows():
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

def display_employee_analysis_tab(df, transformed_df, probabilities, adjusted_probabilities, explanations, features, model):
    """Display the employee analysis tab with detailed insights"""
    # Add back button at the top
    with st.container():
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("← Back to Overview"):
                st.session_state.active_tab = 0
                st.experimental_rerun()
    
    st.markdown('<h1 class="main-header">Employee Analysis</h1>', unsafe_allow_html=True)
    
    # Employee Selection - automatically select the employee from session state
    with st.container():
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
    
    # Check if employee data is cached in session state
    employee_cache_key = f"employee_data_{employee_id}"
    if employee_cache_key not in st.session_state:
        # Employee Details Card
        employee_data = transformed_df.iloc[employee_idx]
        # Get risk score from pre-calculated adjusted probabilities 
        risk_score = adjusted_probabilities[employee_idx]
    
        # Get raw data for display
        raw_data = df.iloc[employee_idx]
        
        # Get the business unit
        bu_columns = [col for col in df.columns if col.startswith('Business_Unit_')]
        employee_bu = next((col.replace('Business_Unit_', '') for col in bu_columns if raw_data[col] == 1), 'Unknown')
        
        # Cache this data
        st.session_state[employee_cache_key] = {
            'employee_data': employee_data,
            'risk_score': risk_score,
            'raw_data': raw_data,
            'employee_bu': employee_bu
        }
    else:
        # Retrieve from cache
        cached_data = st.session_state[employee_cache_key]
        employee_data = cached_data['employee_data']
        risk_score = cached_data['risk_score']
        raw_data = cached_data['raw_data']
        employee_bu = cached_data['employee_bu']
    
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
    with st.container():
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
    
    # Predictive Dashboard - wrap in container for performance
    with st.container():
        st.markdown('<h2 class="sub-header">Predictive Dashboard</h2>', unsafe_allow_html=True)
    
    # Create columns for different views
    col1, col2 = st.columns(2)
    
    with col1:
        # Team/Role Risk Overview
        role_columns = [col for col in df.columns if col.startswith('RoleHistory_')]
        employee_role = next((col.replace('RoleHistory_', '') for col in role_columns if raw_data[col] == 1), 'Unknown')
        
        # Calculate these metrics once and cache them
        role_cache_key = f"role_metrics_{employee_role}"
        if role_cache_key not in st.session_state:
            role_risk = df[[col for col in df.columns if col.startswith('RoleHistory_')]].mean().mean()
            role_changes = max(0, float(raw_data['RoleChanges']))  # Ensure not negative
            peer_attrition = max(0, min(1.0, float(raw_data['TeamAttritionRate'])))  # Clamp between 0 and 1.0
                
            st.session_state[role_cache_key] = {
                'role_risk': role_risk,
                'role_changes': role_changes,
                'peer_attrition': peer_attrition
            }
        else:
            role_metrics = st.session_state[role_cache_key]
            role_risk = role_metrics['role_risk']
            role_changes = role_metrics['role_changes']
            peer_attrition = role_metrics['peer_attrition']
        
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
        # Business Unit Risk - calculate once and cache
        bu_cache_key = f"bu_metrics_{employee_bu}"
        if bu_cache_key not in st.session_state:
            bu_risk = df[df[f'Business_Unit_{employee_bu}'] == 1]['AttritionLabel'].mean() if employee_bu != 'Unknown' else 0
            bu_risk = max(0, min(1.0, float(bu_risk)))  # Clamp between 0 and 1.0
            bu_size = len(df[df[f'Business_Unit_{employee_bu}'] == 1]) if employee_bu != 'Unknown' else 0
            high_risk_members = len(df[(df[f'Business_Unit_{employee_bu}'] == 1) & (adjusted_probabilities >= 0.70)]) if employee_bu != 'Unknown' else 0
                
            st.session_state[bu_cache_key] = {
                'bu_risk': bu_risk,
                'bu_size': bu_size,
                'high_risk_members': high_risk_members
            }
        else:
            bu_metrics = st.session_state[bu_cache_key]
            bu_risk = bu_metrics['bu_risk']
            bu_size = bu_metrics['bu_size']
            high_risk_members = bu_metrics['high_risk_members']
        
        st.markdown(f"""
            <div class="stCard">
                <h4>Business Unit Risk Overview</h4>
                <p><strong>Business Unit:</strong> {employee_bu}</p>
                <p><strong>BU Size:</strong> {bu_size}</p>
                <p><strong>BU Attrition Rate:</strong> {bu_risk:.1%}</p>
                <p><strong>High Risk Members:</strong> {high_risk_members}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Confidence Level and Key Drivers - compute SHAP values only for this employee
    with st.container():
        st.markdown('<h2 class="sub-header">Confidence Level & Key Drivers</h2>', unsafe_allow_html=True)
        
        if explanations is not None:
            # Only get this employee's explanation - more efficient
            employee_explanation = explanations[employee_idx]
            
            # Calculate feature importance based on absolute SHAP values
            impact_df = pd.DataFrame({
                'Feature': features,
                'Impact': employee_explanation,
                'Absolute Impact': np.abs(employee_explanation)
            }).sort_values('Absolute Impact', ascending=False)
            
            # Calculate confidence level based on feature importance distribution
            confidence_score = min(1.0, max(0.0, 1 - (impact_df['Absolute Impact'].std() / (impact_df['Absolute Impact'].mean() + 1e-10))))
            confidence_level = "High" if confidence_score > 0.7 else "Medium" if confidence_score > 0.4 else "Low"
            
            st.markdown(f"""
                <div class="stCard">
                    <h4>Prediction Confidence</h4>
                    <p><strong>Confidence Level:</strong> {confidence_level}</p>
                    <p><strong>Confidence Score:</strong> {confidence_score:.2f}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display key drivers - limit to top 5 for performance
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
    
    # Risk Alerts - wrap in container for performance
    with st.container():
        st.markdown('<h2 class="sub-header">Risk Alerts</h2>', unsafe_allow_html=True)
        
        # Check for role-based risk cluster - calculate once and cache
        role_columns = [col for col in df.columns if col.startswith('RoleHistory_')]
        employee_role = next((col for col in role_columns if raw_data[col] == 1), None)
        
        # Calculate high-risk clusters once and cache
        clusters_cache_key = f"risk_clusters_{employee_id}"
        if clusters_cache_key not in st.session_state:
            role_cluster_found = False
            bu_cluster_found = False
    
        if employee_role:
            # Get count of high-risk role members without materializing the entire dataframe
            role_high_risk_count = ((df[employee_role] == 1) & (adjusted_probabilities >= 0.70)).sum()
            role_cluster_found = role_high_risk_count >= 3
            role_high_risk_count_value = int(role_high_risk_count)
        else:
            role_cluster_found = False
            role_high_risk_count_value = 0
        
        if employee_bu != 'Unknown':
            # Get count of high-risk BU members without materializing the entire dataframe
            bu_high_risk_count = ((df[f'Business_Unit_{employee_bu}'] == 1) & (adjusted_probabilities >= 0.70)).sum()
            bu_cluster_found = bu_high_risk_count >= 5
            bu_high_risk_count_value = int(bu_high_risk_count)
        else:
            bu_cluster_found = False
            bu_high_risk_count_value = 0
        
        st.session_state[clusters_cache_key] = {
            'role_cluster_found': role_cluster_found,
            'role_high_risk_count': role_high_risk_count_value,
            'bu_cluster_found': bu_cluster_found,
            'bu_high_risk_count': bu_high_risk_count_value
        }
    
        if role_cluster_found:
            st.markdown(f"""
                <div style='background-color: #ffebee; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                    <h5>⚠️ High Risk Role Alert</h5>
                    <p><strong>Alert Level:</strong> Critical</p>
                    <p><strong>Details:</strong> {role_high_risk_count} employees in the same role are at high risk of attrition.</p>
                    <p><strong>Recommended Action:</strong> Review role-specific challenges and career progression opportunities.</p>
                </div>
            """, unsafe_allow_html=True)
    
        if bu_cluster_found:
            st.markdown(f"""
                <div style='background-color: #ffebee; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                    <h5>⚠️ High Risk Business Unit Alert</h5>
                    <p><strong>Alert Level:</strong> Critical</p>
                    <p><strong>Details:</strong> {bu_high_risk_count} members of {employee_bu} are at high risk of attrition.</p>
                    <p><strong>Recommended Action:</strong> Review BU leadership and organizational structure.</p>
                </div>
            """, unsafe_allow_html=True)
    
        if not role_cluster_found and not bu_cluster_found:
            st.markdown("""
                <div style='background-color: #e8f5e9; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                    <h5>✓ No Risk Clusters Detected</h5>
                    <p>No significant risk clusters found in this employee's role or business unit.</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Manager's Action Items - wrap in container for performance
    with st.container():
        st.markdown('<h2 class="sub-header">Manager\'s Action Items</h2>', unsafe_allow_html=True)
        
        # Get recommendations for this employee - use consistent thresholds
        risk_level = "High Risk" if risk_score >= 0.70 else "Medium Risk" if risk_score >= 0.50 else "Low Risk"
        
        # Generate recommendations only if not cached
        recs_cache_key = f"recommendations_{employee_id}_{risk_level}"
        if recs_cache_key not in st.session_state:
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
            
            st.session_state[recs_cache_key] = recommendations
        else:
            recommendations = st.session_state[recs_cache_key]
        
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
            # Add default recommendations based on risk level
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

def main():
    """Main function to run the Streamlit app"""
    # Initialize session state for tab selection and risk threshold
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0  # 0 for Overview, 1 for Employee Analysis
    if 'risk_threshold' not in st.session_state:
        st.session_state.risk_threshold = 50  # Medium risk threshold
    if 'selected_employee' not in st.session_state:
        st.session_state.selected_employee = None
    
    # Use st.spinner to show loading state during heavy operations
    with st.spinner("Loading data and model..."):
        # Load data with caching
        df = load_data()
        if df is None:
            st.error("Failed to load data. Please check the data file and try again.")
            return
        
        # Set the index to EmployeeID column
        df.set_index('EmployeeID', inplace=True)
        
        # Transform data with caching
        transformed_df = transform_data(df)
        
        # Define features list based on transformed columns
        features = transformed_df.columns.tolist()
        
        # Validate data
        is_valid, validation_message = validate_data(transformed_df, features)
        if not is_valid:
            st.error(f"Data validation failed: {validation_message}")
            return
        
        # Preprocess data with caching
        try:
            transformed_df = preprocess_data(transformed_df, features)
        except Exception as e:
            st.error(f"Error preprocessing data: {str(e)}")
            return
        
        # Load model and explainer with caching
        model, explainer = load_prediction_resources()
        if model is None or explainer is None:
            st.error("Failed to load prediction resources. Please check the model files and try again.")
            return
        
        # Generate predictions with caching
        try:
            # Check if predictions are cached in session state
            if 'prediction_data' not in st.session_state:
                # Generate new predictions
                probabilities, adjusted_probabilities, explanations = predict_all_employees(transformed_df, model, explainer, features)
                
                # Validate prediction results
                if probabilities is None or explanations is None:
                    st.error("Failed to generate predictions. Please check the model and data.")
                    return
                
                # Store in session state to avoid redundant calculations
                st.session_state.prediction_data = {
                    'probabilities': probabilities,
                    'adjusted_probabilities': adjusted_probabilities,
                    'explanations': explanations
                }
            
            # Get from session state (either newly calculated or previously stored)
            probabilities = st.session_state.prediction_data['probabilities']
            adjusted_probabilities = st.session_state.prediction_data['adjusted_probabilities']
            explanations = st.session_state.prediction_data['explanations']
            
        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")
            return
    
    # Create tabs - use container for better performance
    with st.container():
        tab1, tab2 = st.tabs(["Overview", "Employee Analysis"])
        
        # Display content based on active tab
        if st.session_state.active_tab == 0:
            with tab1:
                display_overview_tab(df, transformed_df, probabilities, adjusted_probabilities, features, explanations)
        else:
            with tab2:
                display_employee_analysis_tab(df, transformed_df, probabilities, adjusted_probabilities, explanations, features, model)

if __name__ == "__main__":
    # Run the main app function
    main()