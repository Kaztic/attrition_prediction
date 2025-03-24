#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Engineer additional features from the employee data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing employee data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional engineered features
    """
    # Create a copy of the DataFrame to avoid warnings
    df_processed = df.copy()
    
    # Calculate team and department level metrics
    df_processed['team_attrition_rate'] = df_processed.groupby('team')['attrition'].transform('mean')
    df_processed['dept_attrition_rate'] = df_processed.groupby('department')['attrition'].transform('mean')
    
    # Calculate manager level metrics
    df_processed['manager_avg_performance'] = df_processed.groupby('manager')['performance_score'].transform('mean')
    df_processed['manager_team_size'] = df_processed.groupby('manager')['employee_id'].transform('count')
    
    # Tenure-based features
    df_processed['promotion_delay'] = np.where(
        (df_processed['tenure'] - df_processed['last_promotion'] > 3) & 
        (df_processed['last_promotion'] > 0), 
        1, 0
    )
    df_processed['never_promoted'] = np.where(df_processed['last_promotion'] == 0, 1, 0)
    
    # Create a composite engagement score
    df_processed['engagement_score'] = (
        df_processed['training_hours'] / 100 * 0.3 + 
        df_processed['survey_satisfaction'] / 5 * 0.4 +
        (5 - df_processed['peer_attrition_last60d']) / 5 * 0.3  # inverse of peer attrition
    )
    
    # Growth opportunity indicators
    df_processed['stagnation_risk'] = np.where(
        (df_processed['role_changes'] == 0) & 
        (df_processed['tenure'] > 2) & 
        (df_processed['performance_score'] >= 4),
        1, 0
    )
    
    # Performance vs. peer comparison (within department)
    df_processed['performance_above_dept_avg'] = np.where(
        df_processed['performance_score'] > df_processed.groupby('department')['performance_score'].transform('mean'),
        1, 0
    )
    
    # Work-life balance indicator
    max_leave_days = df_processed['leave_days_taken'].max()
    df_processed['leave_utilization'] = df_processed['leave_days_taken'] / max_leave_days
    
    # Interaction terms
    df_processed['high_performer_no_promo'] = np.where(
        (df_processed['performance_score'] >= 4) & 
        (df_processed['promotion_delay'] == 1),
        1, 0
    )
    
    # Normalize certain features for model consumption
    # This helps with model convergence and interpretation
    for col in ['training_hours', 'peer_attrition_last60d', 'leave_days_taken']:
        col_max = df_processed[col].max()
        if col_max > 0:  # Avoid division by zero
            df_processed[f'{col}_normalized'] = df_processed[col] / col_max
    
    return df_processed


if __name__ == "__main__":
    # Test the feature engineering when run directly
    try:
        df = pd.read_csv("../data/employee_data.csv")
        processed_df = engineer_features(df)
        print("Original features:", df.columns.tolist())
        print("\nAll features after engineering:", processed_df.columns.tolist())
        print("\nNew features:", set(processed_df.columns) - set(df.columns))
    except FileNotFoundError:
        print("Data file not found. Generate data first using data_generator.py")