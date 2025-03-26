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
        DataFrame containing employee data from Gemini generator
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional engineered features
    """
    # Create a copy of the DataFrame to avoid warnings
    df_processed = df.copy()
    
    # Convert numeric columns
    numeric_columns = [
        'Tenure', 'Promotions', 'LastPromotionYearsAgo', 'PastPerformance',
        'SkillRelevance', 'TrainingParticipation', 'EventActivity',
        'FeedbackScore', 'SentimentScore', 'TeamAttritionRate',
        'LocationChanges', 'RoleChanges', 'WorkLifeBalance',
        'LeavePattern', 'RecognitionCount', 'AwardsReceived'
    ]
    
    for col in numeric_columns:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Calculate career progression metrics
    df_processed['promotion_delay'] = np.where(
        (df_processed['Tenure'] - df_processed['LastPromotionYearsAgo'] > 3) & 
        (df_processed['LastPromotionYearsAgo'] > 0), 
        1, 0
    )
    df_processed['never_promoted'] = np.where(df_processed['LastPromotionYearsAgo'] == 0, 1, 0)
    
    # Create a composite engagement score
    df_processed['engagement_score'] = (
        df_processed['TrainingParticipation'] / df_processed['TrainingParticipation'].max() * 0.3 + 
        df_processed['FeedbackScore'] / 5 * 0.3 +
        df_processed['EventActivity'] / df_processed['EventActivity'].max() * 0.2 +
        df_processed['WorkLifeBalance'] / 5 * 0.2
    )
    
    # Growth opportunity indicators
    df_processed['stagnation_risk'] = np.where(
        (df_processed['RoleChanges'] == 0) & 
        (df_processed['Tenure'] > 2) & 
        (df_processed['PastPerformance'] >= 1.5),
        1, 0
    )
    
    # Performance vs. peer comparison
    df_processed['performance_above_avg'] = np.where(
        df_processed['PastPerformance'] > df_processed['PastPerformance'].mean(),
        1, 0
    )
    
    # Work-life balance indicators
    df_processed['leave_utilization'] = df_processed['LeavePattern'] / df_processed['LeavePattern'].max()
    
    # Career satisfaction indicators
    df_processed['career_satisfaction'] = (
        df_processed['PastPerformance'] / 1.8 * 0.3 +
        df_processed['SkillRelevance'] * 0.2 +
        df_processed['WorkLifeBalance'] / 5 * 0.3 +
        df_processed['SentimentScore'] * 0.2
    )
    
    # Risk factors for attrition
    df_processed['attrition_risk_score'] = (
        (df_processed['TeamAttritionRate'] * 0.3) +
        (df_processed['ManagerAttrition'] * 0.2) +
        (1 - df_processed['WorkLifeBalance'] / 5) * 0.2 +
        (1 - df_processed['career_satisfaction']) * 0.3
    )
    
    # Interaction terms
    df_processed['high_performer_no_promo'] = np.where(
        (df_processed['PastPerformance'] >= 1.5) & 
        (df_processed['promotion_delay'] == 1),
        1, 0
    )
    
    # Normalize certain features for model consumption
    for col in ['TrainingParticipation', 'EventActivity', 'LeavePattern']:
        if col in df_processed.columns:
            col_max = df_processed[col].max()
            if col_max > 0:  # Avoid division by zero
                df_processed[f'{col}_normalized'] = df_processed[col] / col_max
    
    return df_processed


if __name__ == "__main__":
    # Test the feature engineering when run directly
    try:
        df = pd.read_csv("structured_employee_data.csv")
        processed_df = engineer_features(df)
        print("Original features:", df.columns.tolist())
        print("\nAll features after engineering:", processed_df.columns.tolist())
        print("\nNew features:", set(processed_df.columns) - set(df.columns))
    except FileNotFoundError:
        print("Data file not found. Generate data first using data_generator.py")