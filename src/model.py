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
                n_estimators=20,
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
        
        for _, row in feature_importance.iterrows():
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
    predict_proba = explanation_dict['prediction_probability']
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig('shap-value.png', format='png', dpi=600)
    plt.show()

    # Save model
    # model.save_model()
    
    return {
        'model': model,
        'metrics': evaluation_metrics,
        'shap_values': shap_values,
        'feature_importance':feature_importance,
        'predict_proba':predict_proba
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
    
    processed_data = processed_data.drop(['EmployeeID'], axis=1)
    processed_data = processed_data.drop(['ExitReason_None'], axis=1)
    
    result = train_attrition_model(processed_data)
    
    print("Model Evaluation Metrics:")
    
    for metric, value in result['metrics'].items():
        print(f"{metric}: {value}")