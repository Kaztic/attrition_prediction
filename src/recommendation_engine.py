#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def generate_recommendations(employee_data, model_explanation, features):
    """
    Generate personalized retention recommendations based on risk factors.
    
    Parameters:
    -----------
    employee_data : pandas.DataFrame
        Employee data for which to generate recommendations
    model_explanation : numpy.ndarray
        SHAP values for the employee data
    features : list
        List of feature names
        
    Returns:
    --------
    list
        List of recommendation dictionaries
    """
    recommendations = []
    
    # Extract top risk factors based on SHAP values
    risk_factors = extract_risk_factors(model_explanation, features)
    
    # Map risk factors to recommendations
    for feature, importance in risk_factors:
        if importance <= 0:  # Skip factors that decrease attrition risk
            continue
            
        # Generate recommendations based on feature and importance
        if "LastPromotionYearsAgo" in feature and importance > 0.05 :
            recommendations.append({
                "action": "Career progression review",
                "details": "Schedule a career path discussion to explore growth opportunities",
                "priority": "High" if importance > 0.05 else "Medium"
            })
            
        elif "tenure" in feature and importance > 0:
            recommendations.append({
                "action": "Tenure-based recognition",
                "details": "Recognize contributions and discuss long-term vision",
                "priority": "Medium"
            })
            
        elif "PastPerformance" in feature and importance > 0:
            if "AwardsReceived" in feature and importance > :
                recommendations.append({
                    "action": "High performer retention",
                    "details": "Provide special projects, visibility with leadership, and growth opportunities",
                    "priority": "High"
                })
            else:
                recommendations.append({
                    "action": "Performance coaching",
                    "details": "Provide additional support, training, and clear expectations",
                    "priority": "Medium"
                })
                
        elif "engagement" in feature or "training" in feature:
            recommendations.append({
                "action": "Engagement initiatives",
                "details": "Increase learning opportunities and involvement in meaningful projects",
                "priority": "High" if importance > 0.1 else "Medium"
            })
            
        elif "peer_attrition" in feature:
            recommendations.append({
                "action": "Team stability focus",
                "details": "Address team concerns, improve communication, and build team cohesion",
                "priority": "High" if importance > 0.1 else "Medium"
            })
            
        elif "stagnation" in feature:
            recommendations.append({
                "action": "Role enrichment",
                "details": "Provide new challenges, cross-training, or special assignments",
                "priority": "High"
            })
            
        elif "survey_satisfaction" in feature:
            recommendations.append({
                "action": "Satisfaction check-in",
                "details": "Conduct one-on-one meeting to understand concerns and improvement areas",
                "priority": "High" if importance > 0.1 else "Medium"
            })
            
        elif "leave" in feature:
            recommendations.append({
                "action": "Work-life balance review",
                "details": "Check for burnout signs and ensure appropriate workload",
                "priority": "Medium"
            })
    
    # Remove duplicate recommendations
    unique_recommendations = []
    action_set = set()
    
    for rec in recommendations:
        if rec["action"] not in action_set:
            unique_recommendations.append(rec)
            action_set.add(rec["action"])
    
    # Sort by priority
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    sorted_recommendations = sorted(unique_recommendations, key=lambda x: priority_order[x["priority"]])
    
    return sorted_recommendations


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
        
    # Create feature importance pairs
    feature_importance = list(zip(features, shap_values))
    
    # Sort by absolute importance (higher values have more impact)
    sorted_importance = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)
    
    # Return top N factors
    return sorted_importance[:top_n]


def identify_risk_clusters(employee_data, predictions, explanations, features):
    """
    Identify clusters of employees with similar risk patterns.
    
    Parameters:
    -----------
    employee_data : pandas.DataFrame
        Employee data
    predictions : numpy.ndarray
        Attrition probability predictions
    explanations : numpy.ndarray
        SHAP values for the employee data
    features : list
        List of feature names
        
    Returns:
    --------
    list
        List of risk cluster dictionaries
    """
    # Add predictions to employee data
    df = employee_data.copy()
    df['attrition_prob'] = predictions
    
    # Define high risk threshold
    high_risk_threshold = 0.7
    high_risk_employees = df[df['attrition_prob'] >= high_risk_threshold]
    
    clusters = []
    
    # Cluster 1: High performers with stagnation risk
    high_performers = high_risk_employees[
        (high_risk_employees['performance_score'] >= 4) & 
        ((high_risk_employees['stagnation_risk'] == 1) if 'stagnation_risk' in high_risk_employees.columns else True)
    ]
    
    if len(high_performers) > 0:
        clusters.append({
            "name": "High Performers at Risk",
            "count": len(high_performers),
            "avg_risk": high_performers['attrition_prob'].mean(),
            "ids": high_performers['employee_id'].tolist(),
            "actions": [
                "Create clear growth paths with milestones",
                "Provide challenging assignments",
                "Consider for leadership or mentoring roles",
                "Review compensation competitiveness"
            ]
        })
    
    # Cluster 2: Team stability issues
    team_instability = high_risk_employees[
        high_risk_employees['peer_attrition_last60d'] >= 2
    ]
    
    if len(team_instability) > 0:
        # Group by team to find teams with multiple at-risk employees
        team_risks = team_instability.groupby('team')['employee_id'].count().reset_index()
        team_risks = team_risks[team_risks['employee_id'] > 1]
        
        if not team_risks.empty:
            risk_teams = team_risks['team'].tolist()
            team_members = high_risk_employees[high_risk_employees['team'].isin(risk_teams)]
            
            clusters.append({
                "name": "Team Stability Concerns",
                "count": len(team_members),
                "avg_risk": team_members['attrition_prob'].mean(),
                "ids": team_members['employee_id'].tolist(),
                "teams": risk_teams,
                "actions": [
                    "Conduct team health assessment",
                    "Address manager-team dynamics",
                    "Improve team communication",
                    "Consider team-building activities",
                    "Review workload distribution"
                ]
            })
    
    # Cluster 3: Engaged but promotion-delayed employees
    promotion_delayed = high_risk_employees[
        ((high_risk_employees['promotion_delay'] == 1) if 'promotion_delay' in high_risk_employees.columns else 
         (high_risk_employees['tenure'] - high_risk_employees['last_promotion'] > 3)) &
        ((high_risk_employees['engagement_score'] >= 0.6) if 'engagement_score' in high_risk_employees.columns else True)
    ]
    
    if len(promotion_delayed) > 0:
        clusters.append({
            "name": "Engaged but Career-Stalled",
            "count": len(promotion_delayed),
            "avg_risk": promotion_delayed['attrition_prob'].mean(),
            "ids": promotion_delayed['employee_id'].tolist(),
            "actions": [
                "Review promotion criteria and timelines",
                "Create interim growth opportunities",
                "Provide honest career path discussions",
                "Consider role expansion or enrichment",
                "Explore lateral moves for skill diversification"
            ]
        })
    
    return clusters


if __name__ == "__main__":
    # Test the recommendation engine when run directly
    print("Recommendation Engine Module")
    print("Run main.py to use this module within the application")