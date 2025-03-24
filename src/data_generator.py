#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from faker import Faker
import random

def generate_synthetic_data(n_employees=1000, seed=42):
    """
    Generate synthetic employee data for attrition prediction.
    
    Parameters:
    -----------
    n_employees : int
        Number of employee records to generate
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing synthetic employee data
    """
    # Set seeds for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    fake = Faker()
    Faker.seed(seed)
    
    # Create team and manager mappings
    teams = [f"Team-{i}" for i in range(1, 21)]
    departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations', 'IT', 'Customer Support']
    managers = [f"Manager-{i}" for i in range(1, 51)]
    
    # Map managers to departments
    manager_dept = {}
    dept_managers = {dept: [] for dept in departments}
    
    for manager in managers:
        dept = random.choice(departments)
        manager_dept[manager] = dept
        dept_managers[dept].append(manager)
    
    # Generate employee data
    data = []
    
    for i in range(n_employees):
        # Basic employee info
        dept = random.choice(departments)
        manager = random.choice(dept_managers[dept])
        team = random.choice(teams)
        tenure = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                               p=[0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.08, 0.05, 0.05, 0.02])
        
        # Previous performance scores (up to 3 years)
        max_history = min(tenure, 3)
        perf_history = [np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.25, 0.4, 0.2]) 
                        for _ in range(max_history)]
        
        # Current performance
        performance_score = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.25, 0.4, 0.2])
        
        # Last promotion (0 = never promoted)
        max_promo_time = max(0, tenure - 1)
        last_promotion = np.random.choice(list(range(max_promo_time + 1)), 
                                      p=[0.3] + [0.7/max_promo_time] * max_promo_time if max_promo_time > 0 else [1.0])
        
        # More employee attributes
        training_hours = np.random.randint(0, 100)
        role_changes = np.random.randint(0, min(4, tenure))
        survey_satisfaction = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.4, 0.25])
        peer_attrition_last60d = np.random.randint(0, 5)
        leave_days_taken = np.random.randint(0, 25)
        skill_relevance = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.3, 0.3, 0.2])
        
        # Attrition risk factors
        long_tenure_no_promo = 1 if (tenure > 3 and last_promotion == 0) else 0
        recent_perf_drop = 1 if (len(perf_history) >= 2 and perf_history[0] - perf_history[1] < -1) else 0
        high_peer_attrition = 1 if peer_attrition_last60d > 2 else 0
        
        # Determine attrition probability based on risk factors
        attrition_prob = 0.10  # Base probability
        
        if long_tenure_no_promo:
            attrition_prob += 0.15
        if recent_perf_drop:
            attrition_prob += 0.10
        if high_peer_attrition:
            attrition_prob += 0.15
        if survey_satisfaction <= 2:
            attrition_prob += 0.20
        if performance_score >= 4:  # High performers are at risk too
            attrition_prob += 0.05
            
        # Cap probability at 0.85
        attrition_prob = min(0.85, attrition_prob)
        
        # Determine actual attrition
        attrition = np.random.choice([0, 1], p=[1-attrition_prob, attrition_prob])
        
        employee = {
            'employee_id': i+1000,
            'department': dept,
            'team': team,
            'manager': manager,
            'tenure': tenure,
            'performance_score': performance_score,
            'last_promotion': last_promotion,
            'training_hours': training_hours,
            'role_changes': role_changes,
            'survey_satisfaction': survey_satisfaction,
            'peer_attrition_last60d': peer_attrition_last60d,
            'leave_days_taken': leave_days_taken,
            'skill_relevance': skill_relevance,
            'attrition': attrition,
            'attrition_prob': attrition_prob  # Actual probability used (for validation)
        }
        data.append(employee)
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Generate sample data when run directly
    df = generate_synthetic_data(n_employees=100)
    print(df.head())
    print(f"\nData shape: {df.shape}")
    print(f"\nAttrition rate: {df['attrition'].mean():.2f}") 