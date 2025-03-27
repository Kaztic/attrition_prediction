# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import pandas as pd
# import numpy as np
# import pickle
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
# import matplotlib.pyplot as plt
# import shap
# import warnings

# # Suppress shap warnings
# warnings.filterwarnings('ignore')

# def build_model(df, test_size=0.3, random_state=42):
#     """
#     Build and train an attrition prediction model.
    
#     Parameters:
#     -----------
#     df : pandas.DataFrame
#         DataFrame containing processed employee data
#     test_size : float
#         Proportion of data to use for testing
#     random_state : int
#         Random seed for reproducibility
        
#     Returns:
#     --------
#     tuple
#         (model, explainer, X_test, y_test, features)
#     """
#     # Identify features and target
#     exclude_cols = ['employee_id', 'attrition', 'attrition_prob', 'team', 'manager', 'department']
#     features = [col for col in df.columns if col not in exclude_cols]
    
#     # Print features being used
#     print(f"Using {len(features)} features for model training:")
#     for i, feature in enumerate(features, 1):
#         print(f"  {i}. {feature}")
    
#     X = df[features]
#     y = df['attrition']
    
#     # Split data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
#     print(f"\nTraining set: {X_train.shape[0]} samples")
#     print(f"Testing set: {X_test.shape[0]} samples")
    
#     # Train RandomForest model
#     model = RandomForestClassifier(
#         n_estimators=100,
#         max_depth=10,
#         min_samples_split=5,
#         min_samples_leaf=2,
#         random_state=random_state,
#         class_weight='balanced'
#     )
    
#     print("\nTraining model...")
#     model.fit(X_train, y_train)
    
#     # Create SHAP explainer
#     print("Creating SHAP explainer...")
#     explainer = shap.TreeExplainer(model)
    
#     return model, explainer, X_test, y_test, features


# def evaluate_model(model, X_test, y_test):
#     """
#     Evaluate the model performance.
    
#     Parameters:
#     -----------
#     model : sklearn model
#         Trained model
#     X_test : pandas.DataFrame
#         Test features
#     y_test : pandas.Series
#         Test target values
#     """
#     # Make predictions
#     y_pred = model.predict(X_test)
#     y_prob = model.predict_proba(X_test)[:, 1]
    
#     # Print classification report
#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred))
    
#     # Print confusion matrix
#     print("\nConfusion Matrix:")
#     cm = confusion_matrix(y_test, y_pred)
#     print(cm)
    
#     # Calculate ROC AUC
#     roc_auc = roc_auc_score(y_test, y_prob)
#     print(f"\nROC AUC Score: {roc_auc:.4f}")
    
#     # Print feature importance
#     print("\nFeature Importance:")
#     feature_importance = pd.DataFrame({
#         'Feature': X_test.columns,
#         'Importance': model.feature_importances_
#     }).sort_values('Importance', ascending=False)
    
#     for _, row in feature_importance.head(10).iterrows():
#         print(f"  {row['Feature']}: {row['Importance']:.4f}")


# def save_model(model, explainer, features, model_dir='models'):
#     """
#     Save the trained model, explainer, and features list.
    
#     Parameters:
#     -----------
#     model : sklearn model
#         Trained model
#     explainer : shap.TreeExplainer
#         SHAP explainer
#     features : list
#         List of feature names
#     model_dir : str
#         Directory to save model artifacts
#     """
#     # Create directory if it doesn't exist
#     os.makedirs(model_dir, exist_ok=True)
    
#     # Save model
#     with open(os.path.join(model_dir, 'attrition_model.pkl'), 'wb') as f:
#         pickle.dump(model, f)
    
#     # Save explainer
#     with open(os.path.join(model_dir, 'shap_explainer.pkl'), 'wb') as f:
#         pickle.dump(explainer, f)
    
#     # Save feature list
#     with open(os.path.join(model_dir, 'model_features.pkl'), 'wb') as f:
#         pickle.dump(features, f)
    
#     print(f"\nModel artifacts saved to '{model_dir}' directory.")


# def load_model(model_dir='models'):
#     """
#     Load the trained model, explainer, and features list.
    
#     Parameters:
#     -----------
#     model_dir : str
#         Directory with model artifacts
        
#     Returns:
#     --------
#     tuple
#         (model, explainer, features)
#     """
#     # Load model
#     with open(os.path.join(model_dir, 'attrition_model.pkl'), 'rb') as f:
#         model = pickle.load(f)
    
#     # Load explainer
#     with open(os.path.join(model_dir, 'shap_explainer.pkl'), 'rb') as f:
#         explainer = pickle.load(f)
    
#     # Load feature list
#     with open(os.path.join(model_dir, 'model_features.pkl'), 'rb') as f:
#         features = pickle.load(f)
    
#     return model, explainer, features


# def predict_attrition(employee_data, model=None, explainer=None, features=None, model_dir='models'):
#     """
#     Predict attrition probability for employee(s).
    
#     Parameters:
#     -----------
#     employee_data : pandas.DataFrame
#         Employee data to predict attrition for
#     model : sklearn model, optional
#         Trained model (loads from disk if None)
#     explainer : shap.TreeExplainer, optional
#         SHAP explainer (loads from disk if None)
#     features : list, optional
#         List of feature names (loads from disk if None)
#     model_dir : str
#         Directory with model artifacts
        
#     Returns:
#     --------
#     tuple
#         (predictions, probabilities, explanations)
#     """
#     # Load model artifacts if not provided
#     if model is None or explainer is None or features is None:
#         model, explainer, features = load_model(model_dir)
    
#     # Ensure employee_data has all required features
#     for feature in features:
#         if feature not in employee_data.columns:
#             employee_data[feature] = 0  # Default value for missing features
    
#     # Get features in the right order
#     X = employee_data[features]
    
#     # Make predictions
#     predictions = model.predict(X)
#     probabilities = model.predict_proba(X)[:, 1]
    
#     # Generate SHAP explanations
#     try:
#         explanations = explainer.shap_values(X)
#         if isinstance(explanations, list):
#             explanations = explanations[0]  # For binary classification, get first class
#     except Exception as e:
#         print(f"Warning: Could not generate SHAP explanations: {str(e)}")
#         explanations = None
    
#     return predictions, probabilities, explanations


# if __name__ == "__main__":
#     # Test model functionality when run directly
#     try:
#         df = pd.read_csv("../data/processed_employee_data.csv")
#         model, explainer, X_test, y_test, features = build_model(df)
#         evaluate_model(model, X_test, y_test)
#         save_model(model, explainer, features)
        
#         # Test prediction
#         sample = X_test.iloc[0:5]
#         predictions, probabilities, explanations = predict_attrition(sample, model, explainer, features)
#         print("\nSample predictions:")
#         for i, prediction in enumerate(predictions):
#             print(f"  Employee {i+1}: {prediction} (probability of attrition: {probabilities[i]:.4f})")
            
#     except FileNotFoundError:
#         print("Processed data file not found. Process data first using feature_engineering.py")

# src/models/attrition_model.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from typing import Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

import shap
# import lime.lime_tabular

class AttritionPredictionModel:
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the attrition prediction model.
        
        Args:
            model_type (str): Type of model to use
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
    
    def train(self, X, y, feature_names=None):
        """
        Train the attrition prediction model.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Target variable
            feature_names (list): Names of features
        """
        self.feature_names = feature_names
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Model selection and training
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=10,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
        
        self.model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = self.model.predict(X_val)

        # Print feature importance
        print("\nFeature Importance:")
        feature_importance = pd.DataFrame({
            'Feature': X_val.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        for _, row in feature_importance.head(10).iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")
        
        return feature_importance, self._evaluate_model(y_val, y_pred)
    
    def _evaluate_model(self, y_true, y_pred):
        """
        Comprehensive model evaluation.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
        
        Returns:
            Dict of evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        return metrics
    
    def predict(self, X):
        """
        Predict attrition probability.
        
        Args:
            X (np.ndarray): Input features
        
        Returns:
            np.ndarray: Predicted probabilities
        """
        return self.model.predict_proba(X)[:, 1]
    
    def explain_prediction(self, X, instance_index=0, explainer_type='shap'):
        """
        Provide model explanation for a specific prediction.
        
        Args:
            X (np.ndarray): Input features
            instance_index (int): Index of instance to explain
            explainer_type (str): Type of explainability method
        
        Returns:
            Dict of explanation results
        """
        if explainer_type == 'shap':
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            
            return {
                'shap_values': shap_values,
                'base_value': explainer.expected_value,
                'feature_names': self.feature_names,
                'prediction_probability': self.predict(X)
            }
        
        # elif explainer_type == 'lime':
            # explainer = lime.lime_tabular.LimeTabularExplainer(
            #     X, feature_names=self.feature_names, 
            #     class_names=['Retain', 'Attrite']
            # )
            
            # explanation = explainer.explain_instance(
            #     X[instance_index], 
            #     self.model.predict_proba, 
            #     num_features=5
            # )
            
            # return {
            #     'lime_explanation': explanation.as_list(),
            #     'prediction_probability': self.predict(X[instance_index].reshape(1, -1))[0]
            # }
    
    def save_model(self, path='models/attrition_model.pkl'):
        """Save trained model to disk."""
        joblib.dump(self, path)
    
    @classmethod
    def load_model(cls, path='models/attrition_model.pkl'):
        """Load model from disk."""
        return joblib.load(path)

def train_attrition_model(processed_data: Dict[str, Any]):
    """
    Public function to train attrition prediction model.
    
    Args:
        processed_data (Dict): Preprocessed data dictionary
    
    Returns:
        Trained model and evaluation metrics
    """
    model = AttritionPredictionModel()
    y=processed_data['AttritionLabel']
    X=processed_data.drop(['AttritionLabel'], axis=1)
    feature_importance, evaluation_metrics = model.train(
        X, 
        y,
        X.columns
    )
    
    explanation_dict = model.explain_prediction(X)
    shap_values = explanation_dict['shap_values']
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig('shap-value.png', format='png', dpi=600)
    plt.show()

    # Save model
    # model.save_model()
    
    return {
        'model': model,
        'metrics': evaluation_metrics,
        'explanation': explanation_dict
    }

# Example usage
if __name__ == "__main__":
    
    filepath = "./data/preprocessed_employee_data.csv"
    # synthetic_data = generate_employee_dataset()
    processed_data = pd.read_csv(filepath, header=0)

    for column in processed_data.columns:
        if processed_data[column].isnull().any():  # Check if the column has NaN values
            processed_data[column].fillna(processed_data[column].mean(), inplace=True)

    processed_data['AttritionLabel']  = processed_data['AttritionLabel'].astype(int)
    processed_data['ManagerAttrition']  = processed_data['ManagerAttrition'].astype(int)
    print("Must be binary classification!!! = ",processed_data['AttritionLabel'].value_counts())
    processed_data = processed_data.drop(['ExitReason_None'], axis=1)
    
    result = train_attrition_model(processed_data)
    
    print("Model Evaluation Metrics:")
    
    for metric, value in result['metrics'].items():
        print(f"{metric}: {value}")