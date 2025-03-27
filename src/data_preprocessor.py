import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

def setup_logging():
    """Configure logging for the preprocessing script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def preprocess_employee_data(filepath = None, dataframe=None):
    """
    Preprocess employee data with one-hot encoding and standardization.
    
    Args:
        filepath (str): Path to the CSV file
        dataframe (pd.DataFrame): Input employee panda dataframe
    
    Returns:
        tuple: Preprocessed features (X), target variable (y)
    """
    logger = setup_logging()
    
    try:
        # Load the data
        if dataframe.empty:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded dataset with {len(df)} rows")
        elif filepath == None:
            df = dataframe.copy()
            logger.info(f"Loaded dataset with {len(df)} rows")
        else:
            logger.error(f"Must provide either path to dataset or dataframe")
            return None
        
        # Define column types
        numeric_columns = [
            'Tenure', 'Promotions', 'LastPromotionYearsAgo', 
            'PastPerformance', 'SkillRelevance', 'TrainingParticipation', 
            'EventActivity', 'FeedbackScore', 'SentimentScore', 
            'TeamAttritionRate', 'LocationChanges', 'Feedback360', 
            'RoleChanges', 'WorkLifeBalance', 'LeavePattern', 
            'RecognitionCount', 'AwardsReceived'
        ]
        
        categorical_columns = ['RoleHistory', 'ProjectType', 'ExitReason']
        binary_columns = ['ManagerAttrition', 'AttritionLabel']
        
        # Drop non-feature columns first
        columns_to_drop = ['EmployeeID', 'ManagerDialogue', 'EmployeeResponse']
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(columns_to_drop, axis=1, errors='ignore')
        
        # Convert numeric columns
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert binary columns to int, handling missing values
        for col in binary_columns:
            if col in df.columns:
                # First convert to numeric, then fill missing values with 0, then convert to int
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # Identify which columns actually exist in the dataframe
        numerical_features = [col for col in numeric_columns if col in df.columns]
        categorical_features = [col for col in categorical_columns if col in df.columns]
        binary_features = [col for col in binary_columns if col in df.columns]
        
        # Create preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                # Standardize numerical features
                ('num', StandardScaler(), numerical_features),
                # One-hot encode categorical features
                ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_features),
                # Keep binary features as-is
                ('bin', 'passthrough', binary_features)
            ])
        
        # Fit and transform the data
        df_transformed = preprocessor.fit_transform(df)
        
        # Get feature names after transformation
        feature_names = (
            numerical_features + 
            preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist() + 
            binary_features
        )
        
        # Convert to DataFrame for easier handling
        df_processed = pd.DataFrame(
            df_transformed, 
            columns=feature_names, 
            index=df.index
        )
        
        # Logging preprocessing details
        logger.info("Preprocessing Complete")
        logger.info(f"Original features: {len(df.columns)}")
        logger.info(f"Processed features: {len(df_processed.columns)}")
        logger.info(f"Average Attrition Rate: {df_processed['AttritionLabel'].mean() * 100:.2f}%")
        
        return df_processed
    
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

def main():
    # File paths
    input_filepath = './data/processed_employee_data.csv'
    output_features_filepath = './data/preprocessed_employee_data.csv'
    
    try:
        # Preprocess the data
        data_processed = preprocess_employee_data(filepath=input_filepath, dataframe=pd.DataFrame())
        
        # Save processed data
        data_processed.to_csv(output_features_filepath, index=False)
        
        print(f"Processed features saved to: {output_features_filepath}")
    
    except Exception as e:
        print(f"Preprocessing failed: {e}")

if __name__ == "__main__":
    main()