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

# Add the src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import AttritionPredictionModel
from src.recommendation_engine import generate_recommendations, identify_risk_clusters

# Set page configuration
st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    """Load and cache the employee data"""
    try:
        df = pd.read_csv('data/processed_employee_data.csv')
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

def format_risk_score(score):
    """Format the risk score with appropriate styling"""
    # Convert to percentage (0-100)
    percentage = score * 100
    
    if percentage >= 55:
        return f'<span class="risk-high">High Risk ({percentage:.0f}%)</span>'
    elif percentage >= 40:
        return f'<span class="risk-medium">Medium Risk ({percentage:.0f}%)</span>'
    else:
        return f'<span class="risk-low">Low Risk ({percentage:.0f}%)</span>'

def predict_all_employees(df, model, explainer, features):
    """Generate predictions for all employees"""
    try:
        X = df[features]
        probabilities = model.predict_proba(X)[:, 1]
        explanations = explainer.shap_values(X)
        if isinstance(explanations, list):
            explanations = explanations[0]  # For binary classification, get first class
        return probabilities, explanations
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
        return None, None

def display_overview_tab(df, transformed_df, probabilities, features, explanations):
    """Display the overview tab with key metrics and visualizations"""
    st.markdown('<h1 class="main-header">Employee Attrition Analytics</h1>', unsafe_allow_html=True)
    
    # Employee Search
    st.markdown('<h2 class="sub-header">Search Employee</h2>', unsafe_allow_html=True)
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_id = st.text_input("Enter Employee ID", placeholder="e.g., EID770487")
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
    
    # Key Metrics with clickable cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("All Employees", key="all_btn"):
            st.session_state.risk_filter = 'all'
            st.experimental_rerun()
        st.markdown("""
            <div class="metric-card" style="cursor: pointer;">
                <div class="metric-value">{}</div>
                <div class="metric-label">All Employees</div>
            </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        high_risk = (probabilities >= 0.55).sum()
        if st.button("High Risk", key="high_risk_btn"):
            st.session_state.risk_filter = 'high'
            st.experimental_rerun()
        st.markdown("""
            <div class="metric-card" style="cursor: pointer;">
                <div class="metric-value">{}</div>
                <div class="metric-label">High Risk Employees</div>
            </div>
        """.format(high_risk), unsafe_allow_html=True)
    
    with col3:
        medium_risk = ((probabilities >= 0.40) & (probabilities < 0.55)).sum()
        if st.button("Medium Risk", key="medium_risk_btn"):
            st.session_state.risk_filter = 'medium'
            st.experimental_rerun()
        st.markdown("""
            <div class="metric-card" style="cursor: pointer;">
                <div class="metric-value">{}</div>
                <div class="metric-label">Medium Risk Employees</div>
            </div>
        """.format(medium_risk), unsafe_allow_html=True)
    
    with col4:
        low_risk = (probabilities < 0.40).sum()
        if st.button("Low Risk", key="low_risk_btn"):
            st.session_state.risk_filter = 'low'
            st.experimental_rerun()
        st.markdown("""
            <div class="metric-card" style="cursor: pointer;">
                <div class="metric-value">{}</div>
                <div class="metric-label">Low Risk Employees</div>
            </div>
        """.format(low_risk), unsafe_allow_html=True)
    
    # Filter data based on selected risk level
    if st.session_state.risk_filter == 'high':
        filtered_mask = probabilities >= 0.55
    elif st.session_state.risk_filter == 'medium':
        filtered_mask = (probabilities >= 0.40) & (probabilities < 0.55)
    elif st.session_state.risk_filter == 'low':
        filtered_mask = probabilities < 0.40
    else:  # 'all'
        filtered_mask = np.ones(len(df), dtype=bool)
    
    filtered_df = df[filtered_mask]
    filtered_probabilities = probabilities[filtered_mask]
    filtered_transformed_df = transformed_df[filtered_mask]
    
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
    
    # Create a scatter plot
    fig = px.scatter(
        factor_df,
        x='Factor',
        y='Risk Score',
        title=f"Risk Scores vs {selected_factor.replace('_', ' ').title()}",
        labels={'Factor': selected_factor.replace('_', ' ').title(), 'Risk Score': 'Attrition Risk (%)'},
        color='Risk Score',
        color_continuous_scale=['#059669', '#f59e0b', '#dc2626']
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Poppins')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display filtered employees at risk based on the selected factor
    st.markdown('<h2 class="sub-header">Filtered Employees at Risk</h2>', unsafe_allow_html=True)
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
        with col3:
            st.write(f"{selected_factor.replace('_', ' ').title()}: {row[selected_factor]:.2f}")
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
    
    # Employee Selection
    st.markdown('<h2 class="sub-header">Select Employee</h2>', unsafe_allow_html=True)
    # Use session state to maintain selected employee
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
    risk_score = probabilities[employee_idx]
    
    # Get raw data for display
    raw_data = df.iloc[employee_idx]
    
    st.markdown(f"""
        <div class="stCard">
            <h3>Employee Report</h3>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <p><strong>Employee ID:</strong> {employee_id}</p>
                    <p><strong>Tenure:</strong> {raw_data['Tenure']} years</p>
                    <p><strong>Performance Score:</strong> {raw_data['PastPerformance']:.1f}/2.0</p>
                    <p><strong>Training Hours:</strong> {raw_data['TrainingParticipation']:.1f}</p>
                    <p><strong>Role Changes:</strong> {raw_data['RoleChanges']:.1f}</p>
                    <p><strong>Engagement Score:</strong> {(raw_data['EventActivity']/5*100):.1f}%</p>
                </div>
                <div style='text-align: right;'>
                    <h4>Attrition Risk</h4>
                    {format_risk_score(risk_score)}
                    <p style='font-size: 0.8em; color: #666;'>Based on multiple factors including performance, engagement, and career growth</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Feature Impact Analysis
    st.markdown('<h2 class="sub-header">Feature Impact Analysis</h2>', unsafe_allow_html=True)
    if explanations is not None:
        employee_explanation = explanations[employee_idx]
        
        # Calculate feature importance based on absolute SHAP values
        impact_df = pd.DataFrame({
            'Feature': features,
            'Impact': employee_explanation,
            'Absolute Impact': np.abs(employee_explanation)
        }).sort_values('Absolute Impact', ascending=False)
        
        # Create separate dataframes for positive and negative impacts
        positive_impact = impact_df[impact_df['Impact'] > 0].head(5)
        negative_impact = impact_df[impact_df['Impact'] < 0].head(5)
        
        # Create two columns for the plots
        col1, col2 = st.columns(2)
        
        with col1:
            # Plot positive impacts
            fig_positive = px.bar(
                positive_impact,
                x='Impact',
                y='Feature',
                orientation='h',
                title="Top 5 Risk Factors",
                color='Impact',
                color_continuous_scale=['#dc2626', '#f59e0b'],
                labels={'Impact': 'Risk Contribution', 'Feature': 'Factor'}
            )
            fig_positive.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Poppins'),
                showlegend=False
            )
            st.plotly_chart(fig_positive, use_container_width=True)
        
        with col2:
            # Plot negative impacts
            fig_negative = px.bar(
                negative_impact,
                x='Impact',
                y='Feature',
                orientation='h',
                title="Top 5 Protective Factors",
                color='Impact',
                color_continuous_scale=['#059669', '#10b981'],
                labels={'Impact': 'Protection Contribution', 'Feature': 'Factor'}
            )
            fig_negative.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Poppins'),
                showlegend=False
            )
            st.plotly_chart(fig_negative, use_container_width=True)
        
        # Add feature value comparison
        st.markdown('<h3>Feature Value Analysis</h3>', unsafe_allow_html=True)
        feature_values = pd.DataFrame({
            'Feature': features,
            'Value': [employee_data[feature] for feature in features],
            'Impact': employee_explanation,
            'Absolute Impact': np.abs(employee_explanation)
        }).sort_values('Absolute Impact', ascending=False).head(10)
        
        # Calculate relative importance
        max_impact = feature_values['Absolute Impact'].max()
        
        # Define value ranges for each feature
        value_ranges = {
            'performance_score': 'Range: 0.0-2.0',
            'team_attrition_rate': 'Range: 0-100%',
            'dept_attrition_rate': 'Range: 0-100%',
            'leave_days_taken': 'Range: 0-30 days',
            'engagement_score': 'Range: 0-100%',
            'peer_attrition_last60d': 'Range: 0-100%',
            'skill_relevance': 'Range: 0-1.0',
            'training_hours': 'Range: 0-100 hours',
            'manager_team_size': 'Range: 1-50'
        }
        
        # Format values based on feature type
        for idx, row in feature_values.iterrows():
            feature = row['Feature']
            value = row['Value']
            impact = row['Impact']
            relative_importance = (row['Absolute Impact'] / max_impact) * 100
            
            # Format value based on feature type
            if 'score' in feature and 'performance' in feature:
                formatted_value = f"{value:.1f}/2.0"
            elif 'score' in feature or 'rate' in feature:
                formatted_value = f"{value:.1%}"
            elif 'hours' in feature or 'years' in feature:
                formatted_value = f"{value:.1f}"
            else:
                formatted_value = f"{value:.2f}"
            
            # Determine if value is concerning and set appropriate color
            is_high_risk = False
            card_color = '#f5f5f5'  # Default light gray
            
            if 'performance_score' in feature:
                if value < 1.0:
                    is_high_risk = True
                    card_color = '#ffebee'  # Light red
                elif value < 1.2:
                    card_color = '#fff3e0'  # Light orange
                else:
                    card_color = '#e8f5e9'  # Light green
            elif 'rate' in feature:
                if value > 0.2:
                    is_high_risk = True
                    card_color = '#ffebee'  # Light red
                elif value > 0.1:
                    card_color = '#fff3e0'  # Light orange
                else:
                    card_color = '#e8f5e9'  # Light green
            elif 'hours' in feature:
                if value < 20:
                    is_high_risk = True
                    card_color = '#ffebee'  # Light red
                elif value < 30:
                    card_color = '#fff3e0'  # Light orange
                else:
                    card_color = '#e8f5e9'  # Light green
            elif 'engagement_score' in feature:
                if value < 0.4:
                    is_high_risk = True
                    card_color = '#ffebee'  # Light red
                elif value < 0.6:
                    card_color = '#fff3e0'  # Light orange
                else:
                    card_color = '#e8f5e9'  # Light green
            
            # Format impact description with more precise language
            if relative_importance > 80:
                impact_description = "Significantly "
            elif relative_importance > 50:
                impact_description = "Moderately "
            else:
                impact_description = "Slightly "
            
            # Fix contradictions by checking both value and impact
            if ('rate' in feature and value > 0.2) or ('score' in feature and value < 1.0):
                impact_description = "Significantly increases" if impact > 0 else "Significantly decreases"
            elif ('rate' in feature and value > 0.1) or ('score' in feature and value < 1.2):
                impact_description = "Moderately increases" if impact > 0 else "Moderately decreases"
            else:
                impact_description = "Slightly increases" if impact > 0 else "Slightly decreases"
            
            # Add threshold information for better context
            threshold_info = ""
            if 'performance_score' in feature:
                threshold_info = " (Target: >1.2/2.0)"
            elif 'rate' in feature:
                threshold_info = " (Target: <20%)"
            elif 'hours' in feature:
                threshold_info = " (Target: >30 hours)"
            elif 'engagement_score' in feature:
                threshold_info = " (Target: >60%)"
            
            # Get value range for the feature
            range_info = value_ranges.get(feature, "")
            
            st.markdown(f"""
                <div style='background-color: {card_color}; padding: 10px; margin: 5px 0; border-radius: 5px;'>
                    <p><strong>{feature.replace('_', ' ').title()}:</strong> {formatted_value}{threshold_info}</p>
                    <p style='font-size: 0.8em; color: #666;'>{range_info}</p>
                    <p style='font-size: 0.9em; color: {'#dc2626' if impact > 0 else '#059669'};'>
                        {impact_description} attrition risk
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        # Add explanation of feature values
        st.markdown("""
            <div style='background-color: #f8fafc; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                <h4>Understanding Feature Values</h4>
                <p>The color coding indicates the risk level of each feature:</p>
                <ul>
                    <li><span style='color: #dc2626;'>Red</span> - High risk values that need immediate attention</li>
                    <li><span style='color: #f59e0b;'>Orange</span> - Moderate risk values that should be monitored</li>
                    <li><span style='color: #059669;'>Green</span> - Good values that contribute positively</li>
                </ul>
                <p>Target values and ranges are shown for each metric to provide context.</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Recommendations Card
    st.markdown('<h2 class="sub-header">Personalized Recommendations</h2>', unsafe_allow_html=True)
    
    # Get recommendations for this employee
    risk_level = "High Risk" if risk_score >= 0.55 else "Early Warning" if risk_score >= 0.45 else "Low Risk"
    
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
            st.markdown(f"""
                <div style='background-color: {'#ffebee' if rec['priority'] == 'Critical' else '#fff3e0' if rec['priority'] == 'High' else '#e8f5e9'}; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                    <h5>{rec['action']}</h5>
                    <p><strong>Priority:</strong> {rec['priority']}</p>
                    <p><strong>Timeline:</strong> {rec['timeline']}</p>
                    <p>{rec['details']}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No specific recommendations available for this employee at this time.")
        # Add some default recommendations based on risk level
        if risk_level == "High Risk":
            st.markdown("""
                <div style='background-color: #ffebee; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                    <h5>Schedule Immediate Manager Meeting</h5>
                    <p><strong>Priority:</strong> Critical</p>
                    <p><strong>Timeline:</strong> Within 24 hours</p>
                    <p>Arrange an urgent meeting with the employee's manager to discuss concerns and develop an immediate action plan.</p>
                </div>
            """, unsafe_allow_html=True)
        elif risk_level == "Early Warning":
            st.markdown("""
                <div style='background-color: #fff3e0; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                    <h5>Schedule Regular Check-ins</h5>
                    <p><strong>Priority:</strong> High</p>
                    <p><strong>Timeline:</strong> Within 1 week</p>
                    <p>Set up weekly check-ins with the employee to monitor their engagement and address any concerns.</p>
                </div>
            """, unsafe_allow_html=True)

def transform_data(df):
    """Transform the data to match the model's expected features"""
    transformed = pd.DataFrame()
    
    # Convert EmployeeID to string format
    df.index = df.index.astype(str)
    
    # Direct mappings
    transformed['tenure'] = df['Tenure']
    # PastPerformance is already on a scale of 0-2
    transformed['performance_score'] = df['PastPerformance']
    transformed['last_promotion'] = df['LastPromotionYearsAgo']
    transformed['training_hours'] = df['TrainingParticipation']
    transformed['role_changes'] = df['RoleChanges']
    transformed['survey_satisfaction'] = df['FeedbackScore']
    transformed['peer_attrition_last60d'] = df['TeamAttritionRate']
    transformed['leave_days_taken'] = df['LeavePattern']
    transformed['skill_relevance'] = df['SkillRelevance']
    transformed['team_attrition_rate'] = df['TeamAttritionRate']
    transformed['dept_attrition_rate'] = df['ManagerAttrition']  # Using manager attrition as a proxy
    transformed['manager_avg_performance'] = df['Feedback360']  # Using 360 feedback as a proxy
    transformed['manager_team_size'] = 10  # Default value
    transformed['promotion_delay'] = df['LastPromotionYearsAgo']
    transformed['never_promoted'] = (df['Promotions'] == 0).astype(int)
    # Convert EventActivity to percentage (0-100)
    transformed['engagement_score'] = df['EventActivity'] / 5  # Assuming EventActivity is on a scale of 0-5
    transformed['stagnation_risk'] = (df['LastPromotionYearsAgo'] > 3).astype(int)
    transformed['performance_above_dept_avg'] = (df['PastPerformance'] > df['PastPerformance'].mean()).astype(int)
    transformed['leave_utilization'] = df['LeavePattern']
    transformed['high_performer_no_promo'] = ((df['PastPerformance'] > 1.5) & (df['Promotions'] == 0)).astype(int)
    
    # Normalized features
    transformed['training_hours_normalized'] = (df['TrainingParticipation'] - df['TrainingParticipation'].mean()) / df['TrainingParticipation'].std()
    transformed['peer_attrition_last60d_normalized'] = (df['TeamAttritionRate'] - df['TeamAttritionRate'].mean()) / df['TeamAttritionRate'].std()
    transformed['leave_days_taken_normalized'] = (df['LeavePattern'] - df['LeavePattern'].mean()) / df['LeavePattern'].std()
    
    return transformed

def main():
    """Main function to run the Streamlit app"""
    # Initialize session state for tab selection and risk threshold
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0  # 0 for Overview, 1 for Employee Analysis
    if 'risk_threshold' not in st.session_state:
        st.session_state.risk_threshold = 40
    
    # Load data and resources
    df = load_data()
    if df is None:
        st.error("Failed to load data. Please check the data file and try again.")
        return
    
    # Set the index to EmployeeID column
    df.set_index('EmployeeID', inplace=True)
    
    model, explainer = load_prediction_resources()
    if model is None or explainer is None:
        st.error("Failed to load prediction resources. Please check the model files and try again.")
        return
    
    # Transform data to match model's expected features
    transformed_df = transform_data(df)
    
    # Define features
    features = ['tenure', 'performance_score', 'last_promotion', 'training_hours', 'role_changes',
                'survey_satisfaction', 'peer_attrition_last60d', 'leave_days_taken', 'skill_relevance',
                'team_attrition_rate', 'dept_attrition_rate', 'manager_avg_performance', 'manager_team_size',
                'promotion_delay', 'never_promoted', 'engagement_score', 'stagnation_risk',
                'performance_above_dept_avg', 'leave_utilization', 'high_performer_no_promo',
                'training_hours_normalized', 'peer_attrition_last60d_normalized', 'leave_days_taken_normalized']
    
    # Generate predictions
    probabilities, explanations = predict_all_employees(transformed_df, model, explainer, features)
    if probabilities is None or explanations is None:
        st.error("Failed to generate predictions. Please check the model and data.")
        return
    
    # Create tabs
    tab1, tab2 = st.tabs(["Overview", "Employee Analysis"])
    
    # Set active tab based on session state
    active_tab = st.session_state.active_tab
    
    if active_tab == 0:
        display_overview_tab(df, transformed_df, probabilities, features, explanations)
    else:
        display_employee_analysis_tab(df, transformed_df, probabilities, explanations, features, model)

if __name__ == "__main__":
    main()