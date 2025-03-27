#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def generate_recommendations(employee_data, model_explanation, features, risk_level="Early Warning"):
    """
    Generate personalized retention recommendations based on risk factors and risk level.
    
    Parameters:
    -----------
    employee_data : pandas.DataFrame
        Employee data for which to generate recommendations
    model_explanation : numpy.ndarray
        SHAP values for the employee data
    features : list
        List of feature names
    risk_level : str
        Risk level of the employee ("High Risk", "Early Warning", "Low Risk")
        
    Returns:
    --------
    list
        List of recommendation dictionaries with priority and timeline
    """
    recommendations = []
    
    # Extract top risk factors based on SHAP values
    risk_factors = extract_risk_factors(model_explanation, features)
    
    # Get employee's actual values for comparison
    employee_values = employee_data.iloc[0]  # Get first row since we're analyzing one employee
    
    # Risk level specific recommendations
    if risk_level == "High Risk":
        recommendations.append({
            "action": "Immediate Manager Meeting",
            "details": "Schedule an urgent meeting with direct manager to discuss concerns and develop an immediate action plan",
            "priority": "Critical",
            "timeline": "Within 24 hours"
        })
    
    # Generate recommendations based on risk factors and actual values
    for feature, importance in risk_factors:
        if importance <= 0:  # Skip factors that decrease attrition risk
            continue
            
        # Get the actual value for this feature
        actual_value = employee_values[feature]
        
        # Career Growth Related
        if "last_promotion" in feature:
            if actual_value > 3:  # No promotion for more than 3 years
                recommendations.append({
                    "action": "Career Development Plan",
                    "details": f"Create a detailed career development plan with clear milestones. No promotion for {actual_value:.1f} years.",
                    "priority": "High" if importance > 0.1 else "Medium",
                    "timeline": "Within 1 week"
                })
            
        # Performance Related
        elif "performance_score" in feature:
            if actual_value < 0.7:  # Performance below 70%
                recommendations.append({
                    "action": "Performance Improvement Plan",
                    "details": f"Develop a structured performance improvement plan. Current performance score: {actual_value:.1%}",
                    "priority": "High" if importance > 0.1 else "Medium",
                    "timeline": "Within 1 week"
                })
            
        # Training and Development
        elif "training_hours" in feature:
            if actual_value < 20:  # Less than 20 training hours
                recommendations.append({
                    "action": "Skill Development Program",
                    "details": f"Enroll in relevant training programs. Current training hours: {actual_value:.1f}",
                    "priority": "Medium",
                    "timeline": "Within 1 month"
                })
            
        # Role Changes
        elif "role_changes" in feature:
            if actual_value > 3:  # More than 3 role changes
                recommendations.append({
                    "action": "Role Stability Assessment",
                    "details": f"Review role changes. Current number of changes: {actual_value:.1f}",
                    "priority": "Medium",
                    "timeline": "Within 2 weeks"
                })
            
        # Engagement
        elif "engagement_score" in feature:
            if actual_value < 0.6:  # Engagement below 60%
                recommendations.append({
                    "action": "Engagement Improvement Plan",
                    "details": f"Develop initiatives to improve engagement. Current engagement score: {actual_value:.1%}",
                    "priority": "High" if importance > 0.1 else "Medium",
                    "timeline": "Within 2 weeks"
                })
            
        # Work-Life Balance
        elif "leave_days_taken" in feature:
            if actual_value > 15:  # More than 15 days of leave
                recommendations.append({
                    "action": "Work-Life Balance Review",
                    "details": f"Review workload and work-life balance. Leave days taken: {actual_value:.1f}",
                    "priority": "Medium",
                    "timeline": "Within 2 weeks"
                })
            
        # Team Environment
        elif "team_attrition_rate" in feature:
            if actual_value > 0.2:  # Team attrition rate above 20%
                recommendations.append({
                    "action": "Team Stability Initiative",
                    "details": f"Address team stability issues. Current team attrition rate: {actual_value:.1%}",
                    "priority": "High" if importance > 0.1 else "Medium",
                    "timeline": "Within 1 week"
                })
            
        # Skill Relevance
        elif "skill_relevance" in feature:
            if actual_value < 0.7:  # Skill relevance below 70%
                recommendations.append({
                    "action": "Skill Alignment Review",
                    "details": f"Review and align skills with role. Current skill relevance: {actual_value:.1%}",
                    "priority": "Medium",
                    "timeline": "Within 2 weeks"
                })
    
    # Add general recommendations based on risk level
    if risk_level == "High Risk":
        recommendations.append({
            "action": "Retention Package Review",
            "details": "Review and potentially enhance compensation and benefits package",
            "priority": "High",
            "timeline": "Within 1 week"
        })
        
        recommendations.append({
            "action": "Mentorship Program",
            "details": "Assign a senior mentor to provide guidance and support",
            "priority": "High",
            "timeline": "Within 1 week"
        })
    
    elif risk_level == "Early Warning":
        recommendations.append({
            "action": "Regular Check-ins",
            "details": "Set up weekly check-ins with the employee to monitor their engagement and address any concerns",
            "priority": "High",
            "timeline": "Within 1 week"
        })
        
        recommendations.append({
            "action": "Career Growth Discussion",
            "details": "Schedule a career growth discussion to understand aspirations and development needs",
            "priority": "Medium",
            "timeline": "Within 2 weeks"
        })
    
    return recommendations


def extract_risk_factors(shap_values, features, top_n=5):
    """
    Extract top risk factors from SHAP values.
    
    Parameters:
    -----------
    shap_values : numpy.ndarray
        SHAP values for the employee data
    features : list
        List of feature names
    top_n : int
        Number of top factors to extract
        
    Returns:
    --------
    list
        List of (feature, importance) tuples
    """
    # For a single prediction (first dimension is 1)
    if len(shap_values.shape) > 1 and shap_values.shape[0] == 1:
        shap_values = shap_values[0]
    
    # Calculate mean absolute SHAP values for each feature
    if len(shap_values.shape) > 1:
        mean_importance = np.abs(shap_values).mean(axis=0)
    else:
        mean_importance = np.abs(shap_values)
    
    # Create feature importance pairs
    feature_importance = list(zip(features, mean_importance))
    
    # Sort by absolute importance (higher values have more impact)
    sorted_importance = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)
    
    # Return top N factors
    return sorted_importance[:top_n]


def identify_risk_clusters(employee_data, predictions, explanations, features):
    """
    Identify groups of employees with similar risk patterns.
    
    Parameters:
    -----------
    employee_data : pandas.DataFrame
        Employee data
    predictions : numpy.ndarray
        Model predictions
    explanations : numpy.ndarray
        SHAP values
    features : list
        List of feature names
        
    Returns:
    --------
    dict
        Dictionary containing cluster information and patterns
    """
    from sklearn.cluster import KMeans
    
    # Prepare data for clustering
    cluster_data = np.column_stack([
        predictions,
        np.abs(explanations).mean(axis=1)  # Average feature importance
    ])
    
    # Check if we have enough unique points for clustering
    unique_points = np.unique(cluster_data, axis=0)
    n_unique = len(unique_points)
    
    # Adjust number of clusters if needed
    n_clusters = min(3, n_unique)
    
    if n_clusters < 2:
        # If we can't cluster, create a single group
        cluster_patterns = {
            "Cluster_0": {
                "size": len(employee_data),
                "avg_risk_score": float(np.mean(predictions)),
                "risk_level": "High Risk" if np.mean(predictions) > 0.7 else "Early Warning" if np.mean(predictions) > 0.3 else "Low Risk",
                "common_factors": extract_risk_factors(explanations, features)[:3],
                "employee_ids": employee_data.index.tolist()
            }
        }
    else:
        # Perform clustering with adjusted parameters
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,  # Increase number of initializations
            max_iter=300  # Increase maximum iterations
        )
        
        try:
            clusters = kmeans.fit_predict(cluster_data)
        except Exception as e:
            print(f"Warning: Clustering failed: {str(e)}")
            # Fallback to single cluster
            clusters = np.zeros(len(cluster_data), dtype=int)
        
        # Analyze each cluster
        cluster_patterns = {}
        for i in range(n_clusters):
            cluster_mask = clusters == i
            cluster_employees = employee_data[cluster_mask]
            
            if len(cluster_employees) > 0:  # Only process non-empty clusters
                # Calculate cluster characteristics
                avg_risk = float(np.mean(predictions[cluster_mask]))
                cluster_explanations = explanations[cluster_mask]
                top_features = extract_risk_factors(cluster_explanations, features)
                
                cluster_patterns[f"Cluster_{i}"] = {
                    "size": int(np.sum(cluster_mask)),
                    "avg_risk_score": avg_risk,
                    "risk_level": "High Risk" if avg_risk > 0.7 else "Early Warning" if avg_risk > 0.3 else "Low Risk",
                    "common_factors": top_features[:3],
                    "employee_ids": cluster_employees.index.tolist()
                }
    
    return cluster_patterns


if __name__ == "__main__":
    # Test the recommendation engine when run directly
    print("Recommendation Engine Module")
    print("Run main.py to use this module within the application")