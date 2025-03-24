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

# Add the src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import load_model, predict_attrition
from src.recommendation_engine import generate_recommendations, identify_risk_clusters

# Set page configuration
st.set_page_config(
    page_title="Attrition Prediction Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Modern Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Main Layout */
    .main {
        background-color: #ffffff;
    }
    
    /* Headers */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem;
        font-weight: 600;
        color: #34495e;
        margin-bottom: 1.5rem;
        letter-spacing: -0.01em;
    }
    
    /* Risk Indicators */
    .risk-high {
        color: #e74c3c;
        font-weight: 600;
        padding: 0.25rem 0.5rem;
        border-radius: 0.375rem;
        background-color: #fde8e8;
    }
    
    .risk-medium {
        color: #f39c12;
        font-weight: 600;
        padding: 0.25rem 0.5rem;
        border-radius: 0.375rem;
        background-color: #fff3e0;
    }
    
    .risk-low {
        color: #27ae60;
        font-weight: 600;
        padding: 0.25rem 0.5rem;
        border-radius: 0.375rem;
        background-color: #e8f5e9;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #f8fafc;
        border-radius: 0.75rem;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    
    .metric-value {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Insight Boxes */
    .insight-box {
        background-color: #f8fafc;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    
    .insight-box h4 {
        font-family: 'Inter', sans-serif;
        font-size: 1.125rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.75rem;
    }
    
    .insight-box p {
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        color: #64748b;
        line-height: 1.5;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #64748b;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        transition: all 0.2s;
        background-color: #f8fafc;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #2c3e50;
        color: white;
    }
    
    /* Data Tables */
    .dataframe {
        font-family: 'Inter', sans-serif;
        border-radius: 0.75rem;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        background-color: #ffffff;
    }
    
    .dataframe thead th {
        background-color: #f8fafc;
        font-weight: 600;
        color: #2c3e50;
        padding: 1rem;
    }
    
    .dataframe tbody td {
        padding: 1rem;
        color: #64748b;
        background-color: #ffffff;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: #f8fafc;
    }
    
    /* Charts */
    .js-plotly-plot {
        border-radius: 0.75rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        background-color: #ffffff !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    .sidebar .sidebar-content {
        padding: 1.5rem;
    }
    
    /* Buttons */
    .stButton button {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        background-color: #2c3e50;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background-color: #34495e;
    }
    
    /* Select Boxes */
    .stSelectbox select {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        color: #2c3e50;
        padding: 0.5rem;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background-color: #2c3e50;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1rem;
        color: #2c3e50;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .sub-header {
            font-size: 1.5rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            flex-wrap: wrap;
        }
        
        .stTabs [data-baseweb="tab"] {
            flex: 1 1 100%;
            margin-bottom: 0.5rem;
        }
        
        .dataframe {
            overflow-x: auto;
        }
    }
    
    /* Container Responsiveness */
    .stApp {
        max-width: 100%;
        padding: 1rem;
        background-color: #ffffff;
    }
    
    /* Chart Responsiveness */
    .js-plotly-plot {
        width: 100% !important;
        height: auto !important;
    }
    
    /* Table Responsiveness */
    .dataframe-wrapper {
        overflow-x: auto;
        margin: 1rem 0;
    }
    
    /* Sidebar Responsiveness */
    @media (max-width: 768px) {
        .css-1d391kg {
            width: 100% !important;
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            z-index: 1000;
            transform: translateX(-100%);
            transition: transform 0.3s ease-in-out;
            background-color: #f8fafc;
        }
        
        .css-1d391kg[data-testid="stSidebar"][aria-expanded="true"] {
            transform: translateX(0);
        }
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_data():
    """Load the processed employee data"""
    try:
        df = pd.read_csv("data/processed_employee_data.csv")
        return df
    except FileNotFoundError:
        st.error("Processed employee data not found. Please run the data processing step first.")
        return None

@st.cache_resource
def load_prediction_resources():
    """Load the trained model and related resources"""
    try:
        model, explainer, features = load_model()
        return model, explainer, features
    except FileNotFoundError:
        st.error("Model files not found. Please train the model first.")
        return None, None, None

def format_risk_score(score):
    """Format risk score with appropriate color coding"""
    if score >= 0.7:
        return f'<span class="risk-high">{score:.2f}</span>'
    elif score >= 0.4:
        return f'<span class="risk-medium">{score:.2f}</span>'
    else:
        return f'<span class="risk-low">{score:.2f}</span>'

def predict_all_employees(df, model, explainer, features):
    """Make predictions for all employees"""
    # Check if required columns exist
    if not all(feature in df.columns for feature in features):
        missing = [f for f in features if f not in df.columns]
        st.error(f"Missing features in data: {missing}")
        return None, None
    
    # Get features in the right order
    X = df[features]
    
    # Make predictions
    probabilities = model.predict_proba(X)[:, 1]
    
    # Generate SHAP explanations (can be slow for large datasets)
    with st.spinner("Generating explanations..."):
        explanations = explainer.shap_values(X)
    
    return probabilities, explanations

# Main app
def main():
    """Main function to run the dashboard"""
    st.markdown('<p class="main-header">AI-Powered Attrition Prediction</p>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Load model and make predictions
    model, explainer, features = load_prediction_resources()  # Make sure features is loaded here
    if model is None:
        return
    
    # Make predictions
    predictions, probabilities, explanations = predict_attrition(df, model, explainer, features)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Overview", "Employee Analysis", "Recommendations"])
    
    with tab1:
        display_overview_tab(df, probabilities, features, explanations)
    
    with tab2:
        display_employee_analysis_tab(df, probabilities, explanations, features, explainer)
    
    with tab3:
        display_recommendations_tab(df, probabilities, explanations, features)

def display_overview_tab(df, probabilities, features, explanations):
    """Display overview dashboard with key metrics and visualizations"""
    st.markdown('<p class="sub-header">Attrition Risk Overview</p>', unsafe_allow_html=True)
    
    # Calculate risk categories
    high_risk = (probabilities >= 0.7).sum()
    medium_risk = ((probabilities >= 0.4) & (probabilities < 0.7)).sum()
    low_risk = (probabilities < 0.4).sum()
    
    # Summary metrics with enhanced styling
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Total Employees</div>
            </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">High Risk Employees</div>
                <div style="color: #ef4444; font-size: 0.875rem; margin-top: 0.5rem;">{:.1f}% of workforce</div>
            </div>
        """.format(high_risk, high_risk/len(df)*100), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Medium Risk Employees</div>
            </div>
        """.format(medium_risk), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Low Risk Employees</div>
            </div>
        """.format(low_risk), unsafe_allow_html=True)
    
    # Add predictions to dataframe for visualizations
    df_with_pred = df.copy()
    df_with_pred['attrition_risk'] = probabilities
    
    # Risk distribution
    st.markdown('<p class="sub-header">Risk Distribution</p>', unsafe_allow_html=True)
    
    # Create two columns with responsive widths
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Donut chart for risk categories with improved styling
        risk_categories = {
            'High Risk (≥0.7)': high_risk,
            'Medium Risk (0.4-0.69)': medium_risk,
            'Low Risk (<0.4)': low_risk
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=list(risk_categories.keys()),
            values=list(risk_categories.values()),
            hole=0.6,
            marker_colors=['#ef4444', '#f97316', '#22c55e'],
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk distribution by department with improved styling
        dept_risk = df_with_pred.groupby('department')['attrition_risk'].agg(['mean', 'count']).reset_index()
        dept_risk.columns = ['Department', 'Average Risk', 'Employee Count']
        
        fig = px.bar(
            dept_risk, 
            x='Department', 
            y='Average Risk',
            color='Average Risk',
            color_continuous_scale='RdYlGn_r',
            hover_data=['Employee Count'],
            height=400
        )
        
        fig.update_layout(
            yaxis_range=[0, 1],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Department",
            yaxis_title="Average Risk Score",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Team Risk Heatmap with improved styling
    st.markdown('<p class="sub-header">Team Risk Heatmap</p>', unsafe_allow_html=True)
    
    # Calculate team-level metrics
    team_risk = df_with_pred.groupby(['department', 'team'])['attrition_risk'].agg(['mean', 'count']).reset_index()
    team_risk.columns = ['Department', 'Team', 'Average Risk', 'Employee Count']
    
    fig = px.treemap(
        team_risk,
        path=[px.Constant("All Departments"), 'Department', 'Team'],
        values='Employee Count',
        color='Average Risk',
        color_continuous_scale='RdYlGn_r',
        hover_data=['Employee Count', 'Average Risk']
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show risk factors across organization with improved styling
    st.markdown('<p class="sub-header">Top Risk Factors</p>', unsafe_allow_html=True)
    
    if explanations is not None:
        # Calculate feature importance
        mean_importance = np.abs(explanations).mean(axis=0)
        if len(mean_importance) != len(features):
            st.warning(f"Feature mismatch: {len(features)} features vs {len(mean_importance)} importance values")
            return
            
        feature_imp = pd.DataFrame({
            'Feature': features,
            'Importance': mean_importance
        }).sort_values('Importance', ascending=False).head(10)
        
        # Format feature names for better readability
        feature_imp['Feature'] = feature_imp['Feature'].apply(
            lambda x: x.replace('_', ' ').title()
        )
        
        fig = px.bar(
            feature_imp,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=400,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("SHAP explanations not available for feature importance visualization")

def display_employee_analysis_tab(df, probabilities, explanations, features, explainer):
    """Display employee risk analysis with filtering and detailed insights"""
    st.markdown('<p class="sub-header">Employee Risk Analysis</p>', unsafe_allow_html=True)
    
    # Add predictions to dataframe
    df_with_pred = df.copy()
    df_with_pred['attrition_risk'] = probabilities
    
    # Filters in sidebar with enhanced styling
    st.sidebar.markdown("""
        <div style="padding: 1rem; background-color: white; border-radius: 0.75rem; margin-bottom: 1rem;">
            <h3 style="font-family: 'Inter', sans-serif; font-size: 1.25rem; color: #1e293b; margin-bottom: 1rem;">Filters</h3>
    """, unsafe_allow_html=True)
    
    # Department filter
    all_depts = ["All Departments"] + sorted(df['department'].unique().tolist())
    selected_dept = st.sidebar.selectbox("Department", all_depts)
    
    # Team filter (dependent on department)
    if selected_dept != "All Departments":
        dept_teams = ["All Teams"] + sorted(df[df['department'] == selected_dept]['team'].unique().tolist())
        selected_team = st.sidebar.selectbox("Team", dept_teams)
    else:
        selected_team = "All Teams"
    
    # Risk level filter
    risk_threshold = st.sidebar.slider("Minimum Risk Score", 0.0, 1.0, 0.5, 0.05)
    
    # Apply filters
    filtered_df = df_with_pred.copy()
    
    if selected_dept != "All Departments":
        filtered_df = filtered_df[filtered_df['department'] == selected_dept]
        
    if selected_team != "All Teams":
        filtered_df = filtered_df[filtered_df['team'] == selected_team]
        
    filtered_df = filtered_df[filtered_df['attrition_risk'] >= risk_threshold]
    
    # Sort by risk (highest first)
    filtered_df = filtered_df.sort_values('attrition_risk', ascending=False)
    
    # Show filter results
    if len(filtered_df) > 0:
        st.write(f"Showing {len(filtered_df)} employees with risk score ≥ {risk_threshold:.2f}")
        
        # Format the risk score column with colors
        filtered_df['formatted_risk'] = filtered_df['attrition_risk'].apply(
            lambda x: format_risk_score(x)
        )
        
        # Select columns to display
        display_cols = ['employee_id', 'department', 'team', 'manager', 'tenure', 
                       'performance_score', 'formatted_risk']
        
        # Convert to HTML to support the formatted risk column
        html_table = filtered_df[display_cols].to_html(
            escape=False, 
            index=False,
            columns=['employee_id', 'department', 'team', 'tenure', 'performance_score', 'formatted_risk'],
            formatters={'formatted_risk': lambda x: x}
        )
        st.write(html_table, unsafe_allow_html=True)
        
        # Employee selection for detailed analysis
        st.subheader("Employee Deep Dive")
        selected_employee_id = st.selectbox(
            "Select an employee to analyze",
            options=filtered_df['employee_id'].tolist(),
            format_func=lambda x: f"Employee #{x}"
        )
        
        if selected_employee_id:
            employee_idx = df_with_pred[df_with_pred['employee_id'] == selected_employee_id].index[0]
            employee_data = df_with_pred.loc[employee_idx]
            
            # Display employee details and risk factors
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("### Employee Details")
                st.write(f"**ID:** {selected_employee_id}")
                st.write(f"**Department:** {employee_data['department']}")
                st.write(f"**Team:** {employee_data['team']}")
                st.write(f"**Manager:** {employee_data['manager']}")
                st.write(f"**Tenure:** {employee_data['tenure']} years")
                st.write(f"**Performance:** {employee_data['performance_score']}/5")
                st.write(f"**Last Promotion:** {employee_data['last_promotion']} years ago")
                
                # Risk score with color coding
                risk_score = employee_data['attrition_risk']
                risk_html = format_risk_score(risk_score)
                st.write(f"**Risk Score:** {risk_html}", unsafe_allow_html=True)
            
            with col2:
                st.write("### Risk Factors")
                
                if explanations is not None:
                    # Get SHAP values for the selected employee
                    employee_shap_values = explanations[employee_idx]
                    
                    # Create temp figure for SHAP plot
                    fig, ax = plt.subplots(figsize=(10, 5))
                    shap.force_plot(
                        base_value=explainer.expected_value[0],  # Take first class expected value
                        shap_values=employee_shap_values,
                        features=employee_data[features],
                        feature_names=features,
                        matplotlib=True,
                        show=False,
                        text_rotation=45
                    )
                    
                    # Display the figure
                    st.pyplot(fig)
                    
                    # Extract top risk factors
                    feature_importance = list(zip(features, employee_shap_values))
                    sorted_importance = sorted(feature_importance, key=lambda x: -x[1])  # Sort by positive impact on risk
                    
                    # Display top risk factors as a table
                    risk_factors_df = pd.DataFrame(sorted_importance, columns=['Factor', 'Impact'])
                    risk_factors_df = risk_factors_df[risk_factors_df['Impact'] > 0].head(5)  # Only positive factors
                    
                    if not risk_factors_df.empty:
                        st.write("#### Top Risk Drivers")
                        
                        # Format factor names for readability
                        risk_factors_df['Factor'] = risk_factors_df['Factor'].apply(
                            lambda x: x.replace('_', ' ').title()
                        )
                        
                        # Display as a bar chart
                        fig = px.bar(
                            risk_factors_df,
                            x='Impact',
                            y='Factor',
                            orientation='h',
                            color='Impact',
                            color_continuous_scale='Reds'
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("SHAP explanations not available for this employee.")
    else:
        st.write("No employees match the selected filters.")

def display_recommendations_tab(df, probabilities, explanations, features):
    """Display recommendations for retention strategies"""
    st.markdown('<p class="sub-header">Retention Recommendations</p>', unsafe_allow_html=True)
    
    # Add predictions to dataframe
    df_with_pred = df.copy()
    df_with_pred['attrition_risk'] = probabilities
    
    # Generate risk clusters
    with st.spinner("Identifying risk patterns..."):
        risk_clusters = identify_risk_clusters(df_with_pred, probabilities, explanations, features)
    
    # Display risk clusters with enhanced styling
    st.subheader("Risk Clusters")
    
    if risk_clusters:
        for i, cluster in enumerate(risk_clusters):
            st.markdown(f"""
                <div class="insight-box">
                    <h4>{cluster['name']}</h4>
                    <p style="color: #64748b; margin-bottom: 1rem;">
                        {cluster['count']} employees • Average Risk: {cluster['avg_risk']:.2f}
                    </p>
                    <div style="margin-bottom: 1rem;">
                        <h5 style="color: #1e293b; margin-bottom: 0.5rem;">Recommended Actions:</h5>
                        <ul style="margin: 0; padding-left: 1.5rem;">
                            {''.join([f'<li style="color: #64748b; margin-bottom: 0.5rem;">{action}</li>' for action in cluster['actions']])}
                        </ul>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Show employees in this cluster
            if 'ids' in cluster:
                st.markdown("""
                    <div class="insight-box">
                        <h4>Employees in this Cluster</h4>
                """, unsafe_allow_html=True)
                
                cluster_employees = df_with_pred[df_with_pred['employee_id'].isin(cluster['ids'])]
                cluster_employees = cluster_employees[['employee_id', 'department', 'team', 'attrition_risk']]
                cluster_employees = cluster_employees.sort_values('attrition_risk', ascending=False)
                st.dataframe(cluster_employees)
            
            st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
    else:
        st.write("No significant risk clusters identified.")
    
    # Individual recommendation tool
    st.subheader("Individual Recommendation Tool")
    
    # Filter to high-risk employees
    high_risk_df = df_with_pred[df_with_pred['attrition_risk'] >= 0.6]
    
    if not high_risk_df.empty:
        selected_employee_id = st.selectbox(
            "Select a high-risk employee",
            options=high_risk_df['employee_id'].tolist(),
            format_func=lambda x: f"Employee #{x} ({high_risk_df[high_risk_df['employee_id']==x]['attrition_risk'].values[0]:.2f} risk)"
        )
        
        if selected_employee_id:
            employee_idx = df_with_pred[df_with_pred['employee_id'] == selected_employee_id].index[0]
            employee_data = df_with_pred.loc[employee_idx:employee_idx]
            
            # Generate personalized recommendations
            employee_explanation = explanations[0][employee_idx:employee_idx+1]
            recommendations = generate_recommendations(employee_data, employee_explanation, features)
            
            # Display recommendations
            if recommendations:
                for i, rec in enumerate(recommendations):
                    priority_color = {
                        "High": "#D32F2F",
                        "Medium": "#F57C00",
                        "Low": "#388E3C"
                    }
                    
                    st.markdown(
                        f"""
                        <div class="insight-box" style="border-left: 4px solid {priority_color[rec['priority']]}">
                            <h4>{rec['action']} <span style="color:{priority_color[rec['priority']]}">({rec['priority']} Priority)</span></h4>
                            <p>{rec['details']}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            else:
                st.write("No specific recommendations generated.")
    else:
        st.write("No high-risk employees identified.")
    
    # General retention strategies
    with st.expander("General Retention Strategies"):
        st.write("""
        ### Organization-Wide Strategies
        
        1. **Regular career development conversations** - Ensure managers have structured career conversations with team members at least quarterly.
        
        2. **Skill development programs** - Provide learning opportunities aligned with both organizational needs and employee interests.
        
        3. **Recognition programs** - Implement both formal and informal ways to recognize employee contributions.
        
        4. **Flexible work arrangements** - Where possible, offer flexibility in work location and hours.
        
        5. **Competitive compensation reviews** - Regularly benchmark compensation against market rates.
        
        6. **Internal mobility programs** - Create pathways for employees to explore new roles within the organization.
        
        7. **Team building and engagement** - Foster a sense of belonging and purpose through team activities.
        
        8. **Manager training** - Equip managers with skills to support employee growth and retention.
        """)

if __name__ == "__main__":
    main()