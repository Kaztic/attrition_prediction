import pandas as pd
from transformers import pipeline
from typing import List, Dict, Union

from src.feature_mapping_recommendations import feature_keyword_mapping, retention_strategies

class ContextualRetentionRecommender:
    def __init__(self):
        """
        Initialize recommendation engine with contextual strategies.
        """
        self.sentiment_model = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        # Feature to keywords mapping
        self.feature_keyword_mapping = feature_keyword_mapping 
        
        # Retention strategies
        self.retention_strategies = retention_strategies
    
    def extract_contextual_keywords(self, dialogues: Dict[str, List[str]]) -> List[str]:
        """
        Extract meaningful keywords from dialogues.
        
        Args:
            dialogues (Dict[str, List[str]]): Dialogue texts
        
        Returns:
            List of extracted keywords
        """
        keywords = []
        # for dial_list in dialogues.values():
        for dialogue in dialogues.values():
            words = dialogue.lower().split()
            keywords.extend([
                word for word in words 
                if len(word) > 3 and word not in ['this', 'that', 'with', 'from']
            ])
        return list(set(keywords))
    
    def identify_relevant_features(self, keywords: List[str]) -> List[str]:
        """
        Identify relevant features based on keywords.
        
        Args:
            keywords (List[str]): Extracted keywords
        
        Returns:
            List of relevant features
        """
        relevant_features = []
        for feature, feature_keywords in self.feature_keyword_mapping.items():
            # Check if any of the feature keywords are in the dialogue keywords
            if any(fk in keywords for fk in feature_keywords):
                relevant_features.append(feature)
        
        return relevant_features
    
    def generate_personalized_recommendation(
        self, 
        dialogues: Dict[str, List[str]], 
        feature_importance: List[Dict],
        attrition_prob: float
    ) -> Dict[str, Union[str, float, List[Dict]]]:
        """
        Generate a contextually relevant retention recommendation.
        
        Args:
            dialogues (Dict[str, List[str]]): Dialogue texts
            feature_importance (List[Dict]): Model's feature importance
            attrition_prob (float): Predicted attrition probability
        
        Returns:
            Dict with personalized retention recommendation
        """
        # Extract keywords
        keywords = self.extract_contextual_keywords(dialogues)
        
        # Identify relevant features
        relevant_features = self.identify_relevant_features(keywords)
        
        # If no relevant features found, use all features
        if not relevant_features:
            relevant_features = [f['feature'] for f in feature_importance]
        
        # Determine recommendation based on attrition probability and relevant features
        if attrition_prob > 0.7:
            # High risk scenario - prioritize most critical feature
            selected_feature = relevant_features[0]
        elif attrition_prob > 0.4:
            # Moderate risk - choose second most relevant feature
            selected_feature = relevant_features[1] if len(relevant_features) > 1 else relevant_features[0]
        else:
            # Low risk - choose least critical feature for proactive development
            selected_feature = relevant_features[-1]
        
        # Select recommendation strategy
        recommendation = self.retention_strategies.get(
            selected_feature, 
            ["Develop a personalized professional growth strategy"]
        )[0]
        
        return {
            'selected_feature': selected_feature,
            'keywords': keywords,
            'relevant_features': relevant_features,
            'personalized_recommendation': recommendation,
            'attrition_probability': attrition_prob,
            'priority': 'High' if attrition_prob > 0.7 else 'Medium'
        }

def get_employee_retention_recommendation(
    dialogues: Dict[str, List[str]],
    feature_importance: List[Dict],
    attrition_prob: float
) -> Dict:
    """
    Generate a contextually relevant retention recommendation.
    """
    recommender = ContextualRetentionRecommender()
    return recommender.generate_personalized_recommendation(
        dialogues, feature_importance, attrition_prob
    )

# Example usage
if __name__ == "__main__":

    from src.model import train_attrition_model

    # # Simulated model explanation data
    # model_explanation = {
    #     'prediction_probability': 0.75,
    #     'feature_importance': [
    #         {'feature': 'tenure', 'importance': 0.8},
    #         {'feature': 'performance_score', 'importance': 0.6},
    #         {'feature': 'skills_development', 'importance': 0.5}
    #     ]
    # }
    
    # # Example dialogue data
    example_dialogues = {
        'manager_dialogue': [
            "You're doing great work, but we need to discuss your career development.",
            "I appreciate your contributions, but I sense you're looking for more challenges."
        ]
    }

    preprocessed_filepath = "./data/preprocessed_employee_data.csv"
    orig_filepath="./data/processed_employee_data.csv"
    # synthetic_data = generate_employee_dataset()
    preprocessed_data = pd.read_csv(preprocessed_filepath, header=0)
    orig_data= pd.read_csv(orig_filepath, header=0)

    for column in preprocessed_data.columns:
        if preprocessed_data[column].isnull().any():  # Check if the column has NaN values
            preprocessed_data[column].fillna(preprocessed_data[column].mean(), inplace=True)

    preprocessed_data['AttritionLabel']  = preprocessed_data['AttritionLabel'].astype(int)
    preprocessed_data['ManagerAttrition']  = preprocessed_data['ManagerAttrition'].astype(int)
    print("Must be binary classification!!! = ",preprocessed_data['AttritionLabel'].value_counts())
    
    preprocessed_data = preprocessed_data.drop(['EmployeeID'], axis=1)
    preprocessed_data = preprocessed_data.drop(['ExitReason_None'], axis=1)
    
    result = train_attrition_model(preprocessed_data)

    feat_dict = result['feature_importance'].to_dict()
    feat_imp_dict = [
    {'feature': feature, 'importance': importance}
    for idx, feature in feat_dict['Feature'].items()
    for importance in [feat_dict['Importance'][idx]]]

    employee_idx = 1

    print(orig_data.loc[employee_idx, ['ManagerDialogue', 'EmployeeResponse']].to_dict(), "\n\n\n", example_dialogues, "\n\n\n")
    # Get personalized recommendation
    retention_recommendation = get_employee_retention_recommendation(
        orig_data.loc[employee_idx, ['ManagerDialogue', 'EmployeeResponse']].to_dict(), 
        feat_imp_dict, 
        result['predict_proba'][employee_idx]
    )

    # Print results
    from pprint import pprint
    pprint(retention_recommendation)