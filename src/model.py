#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import shap
import warnings

# Suppress shap warnings
warnings.filterwarnings('ignore')

def build_model(df, test_size=0.3, random_state=42):
    """
    Build and train an attrition prediction model.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing processed employee data
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (model, explainer, X_test, y_test, features)
    """
    # Identify features and target
    exclude_cols = ['employee_id', 'attrition', 'attrition_prob', 'team', 'manager', 'department']
    features = [col for col in df.columns if col not in exclude_cols]
    
    # Print features being used
    print(f"Using {len(features)} features for model training:")
    for i, feature in enumerate(features, 1):
        print(f"  {i}. {feature}")
    
    X = df[features]
    y = df['attrition']
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Train RandomForest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight='balanced'
    )
    
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
    # Create SHAP explainer
    print("Creating SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    
    return model, explainer, X_test, y_test, features


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target values
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC AUC Score: {roc_auc:.4f}")
    
    # Print feature importance
    print("\nFeature Importance:")
    feature_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for _, row in feature_importance.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")


def save_model(model, explainer, features, model_dir='models'):
    """
    Save the trained model, explainer, and features list.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    explainer : shap.TreeExplainer
        SHAP explainer
    features : list
        List of feature names
    model_dir : str
        Directory to save model artifacts
    """
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    with open(os.path.join(model_dir, 'attrition_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    # Save explainer
    with open(os.path.join(model_dir, 'shap_explainer.pkl'), 'wb') as f:
        pickle.dump(explainer, f)
    
    # Save feature list
    with open(os.path.join(model_dir, 'model_features.pkl'), 'wb') as f:
        pickle.dump(features, f)
    
    print(f"\nModel artifacts saved to '{model_dir}' directory.")


def load_model(model_dir='models'):
    """
    Load the trained model, explainer, and features list.
    
    Parameters:
    -----------
    model_dir : str
        Directory with model artifacts
        
    Returns:
    --------
    tuple
        (model, explainer, features)
    """
    # Load model
    with open(os.path.join(model_dir, 'attrition_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    # Load explainer
    with open(os.path.join(model_dir, 'shap_explainer.pkl'), 'rb') as f:
        explainer = pickle.load(f)
    
    # Load feature list
    with open(os.path.join(model_dir, 'model_features.pkl'), 'rb') as f:
        features = pickle.load(f)
    
    return model, explainer, features


def predict_attrition(employee_data, model=None, explainer=None, features=None, model_dir='models'):
    """
    Predict attrition probability for employee(s).
    
    Parameters:
    -----------
    employee_data : pandas.DataFrame
        Employee data to predict attrition for
    model : sklearn model, optional
        Trained model (loads from disk if None)
    explainer : shap.TreeExplainer, optional
        SHAP explainer (loads from disk if None)
    features : list, optional
        List of feature names (loads from disk if None)
    model_dir : str
        Directory with model artifacts
        
    Returns:
    --------
    tuple
        (predictions, probabilities, explanations)
    """
    # Load model artifacts if not provided
    if model is None or explainer is None or features is None:
        model, explainer, features = load_model(model_dir)
    
    # Ensure employee_data has all required features
    for feature in features:
        if feature not in employee_data.columns:
            employee_data[feature] = 0  # Default value for missing features
    
    # Get features in the right order
    X = employee_data[features]
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    # Generate SHAP explanations
    try:
        explanations = explainer.shap_values(X)
        if isinstance(explanations, list):
            explanations = explanations[0]  # For binary classification, get first class
    except Exception as e:
        print(f"Warning: Could not generate SHAP explanations: {str(e)}")
        explanations = None
    
    return predictions, probabilities, explanations


if __name__ == "__main__":
    # Test model functionality when run directly
    try:
        df = pd.read_csv("../data/processed_employee_data.csv")
        model, explainer, X_test, y_test, features = build_model(df)
        evaluate_model(model, X_test, y_test)
        save_model(model, explainer, features)
        
        # Test prediction
        sample = X_test.iloc[0:5]
        predictions, probabilities, explanations = predict_attrition(sample, model, explainer, features)
        print("\nSample predictions:")
        for i, prediction in enumerate(predictions):
            print(f"  Employee {i+1}: {prediction} (probability of attrition: {probabilities[i]:.4f})")
            
    except FileNotFoundError:
        print("Processed data file not found. Process data first using feature_engineering.py")