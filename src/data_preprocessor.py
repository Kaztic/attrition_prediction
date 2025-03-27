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
        
        # Store BU and Region columns if they exist
        bu_column = None
        region_column = None
        if 'Business_Unit' in df.columns:
            bu_column = df['Business_Unit'].copy()
            categorical_columns = ['Business_Unit']
        if 'Region' in df.columns:
            region_column = df['Region'].copy()
            categorical_columns = ['Region']
        
        # Apply data constraints
        # Cap attrition rates at 100%
        if 'TeamAttritionRate' in df.columns:
            df['TeamAttritionRate'] = df['TeamAttritionRate'].clip(0, 100) / 100
        if 'DeptAttritionRate' in df.columns:
            df['DeptAttritionRate'] = df['DeptAttritionRate'].clip(0, 100) / 100
            
        # Cap performance scores at 5.0
        if 'PastPerformance' in df.columns:
            df['PastPerformance'] = df['PastPerformance'].clip(0, 5)
            
        # Cap training hours at 100
        if 'TrainingParticipation' in df.columns:
            df['TrainingParticipation'] = df['TrainingParticipation'].clip(0, 100)
            
        # Cap leave days at 30
        if 'LeavePattern' in df.columns:
            df['LeavePattern'] = df['LeavePattern'].clip(0, 30)
            
        # Cap engagement scores at 100%
        if 'FeedbackScore' in df.columns:
            df['FeedbackScore'] = df['FeedbackScore'].clip(0, 100) / 100
            
        # Cap sentiment scores at 1.0
        if 'SentimentScore' in df.columns:
            df['SentimentScore'] = df['SentimentScore'].clip(-1, 1)
            
        # Normalize role changes based on tenure
        if 'RoleChanges' in df.columns and 'Tenure' in df.columns:
            df['RoleChanges'] = df['RoleChanges'] / df['Tenure'].clip(1)  # Avoid division by zero
            
        # Normalize training hours based on tenure
        if 'TrainingParticipation' in df.columns and 'Tenure' in df.columns:
            df['TrainingParticipation'] = df['TrainingParticipation'] / df['Tenure'].clip(1)
            
        # Create composite engagement score
        engagement_components = []
        if 'FeedbackScore' in df.columns:
            engagement_components.append(df['FeedbackScore'])
        if 'SentimentScore' in df.columns:
            engagement_components.append((df['SentimentScore'] + 1) / 2)  # Normalize to 0-1
        if 'TrainingParticipation' in df.columns:
            engagement_components.append(df['TrainingParticipation'] / 100)
            
        if engagement_components:
            df['EngagementScore'] = pd.concat(engagement_components, axis=1).mean(axis=1)
        
        # Define column types
        numeric_columns = [
            'Tenure', 'Promotions', 'LastPromotionYearsAgo', 
            'PastPerformance', 'SkillRelevance', 'TrainingParticipation', 
            'EventActivity', 'FeedbackScore', 'SentimentScore', 
            'TeamAttritionRate', 'LocationChanges', 'Feedback360', 
            'RoleChanges', 'WorkLifeBalance', 'LeavePattern', 
            'RecognitionCount', 'AwardsReceived', 'EngagementScore'
        ]
        
        categorical_columns = ['RoleHistory', 'ProjectType', 'ExitReason']
        binary_columns = ['ManagerAttrition', 'AttritionLabel']
        
        # Handle BU and Region separately to ensure they are preserved
        if 'Business_Unit' in df.columns:
            bu_column = df['Business_Unit'].copy()
            categorical_columns.append('Business_Unit')
        if 'Region' in df.columns:
            region_column = df['Region'].copy()
            categorical_columns.append('Region')
        
        # Drop non-feature columns first
        columns_to_drop = ['ManagerDialogue', 'EmployeeResponse']
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(columns_to_drop, axis=1, errors='ignore')
        
        # Set EmployeeID as index if it exists
        if 'EmployeeID' in df.columns:
            df.set_index('EmployeeID', inplace=True)
        
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
        
        # Add back BU and Region columns if they exist
        if bu_column is not None:
            df_processed['BU'] = bu_column  # Rename Business_Unit to BU for consistency
        if region_column is not None:
            df_processed['Region'] = region_column
        
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
        data_processed.to_csv(output_features_filepath, index=True)
        
        print(f"Processed features saved to: {output_features_filepath}")
    
    except Exception as e:
        print(f"Preprocessing failed: {e}")

if __name__ == "__main__":
    main()