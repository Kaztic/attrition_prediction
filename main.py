#!/usr/bin/env python
# -*- coding: utf-8 -*-
try:
    import IPython
except ImportError:
    # Silently continue without IPython
    pass

import argparse
import pandas as pd
import os
import sys

from src.data_generator import generate_synthetic_data
from src.feature_engineering import engineer_features
from src.model import build_model, save_model, evaluate_model

def main(args):
    """Main function to run the attrition prediction system."""
    # Generate or load data
    if args.generate_data:
        print("Generating synthetic data...")
        df = generate_synthetic_data(n_employees=args.n_employees)
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/employee_data.csv", index=False)
        print(f"Data saved to data/employee_data.csv")
    else:
        print("Loading existing data...")
        try:
            df = pd.read_csv("data/employee_data.csv")
        except FileNotFoundError:
            print("Error: data/employee_data.csv not found. Generate data first with --generate-data")
            return
    
    # Feature engineering
    if args.process_data:
        print("Engineering features...")
        df = engineer_features(df)
        df.to_csv("data/processed_employee_data.csv", index=False)
        print(f"Processed data saved to data/processed_employee_data.csv")
    
    # Build and save model
    if args.train_model:
        print("Training model...")
        try:
            if args.process_data:
                # Use the data we just processed
                pass
            else:
                # Try to load processed data
                df = pd.read_csv("data/processed_employee_data.csv")
        except FileNotFoundError:
            print("Error: processed_employee_data.csv not found. Process data first with --process-data")
            return
            
        model, explainer, X_test, y_test, features = build_model(df)
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        save_model(model, explainer, features)
        
        # Evaluate model
        print("\nModel Evaluation:")
        evaluate_model(model, X_test, y_test)
    
    # Run dashboard
    if args.run_dashboard:
        print("Launching dashboard... (use Ctrl+C to stop)")
        os.system("streamlit run app/app.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Attrition Prediction System")
    parser.add_argument("--generate-data", action="store_true", help="Generate new synthetic data")
    parser.add_argument("--n-employees", type=int, default=1000, help="Number of synthetic employees to generate")
    parser.add_argument("--process-data", action="store_true", help="Process and engineer features")
    parser.add_argument("--train-model", action="store_true", help="Train and save the model")
    parser.add_argument("--run-dashboard", action="store_true", help="Launch the Streamlit dashboard")
    
    args = parser.parse_args()
    
    # If no flags are provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(1)
        
    main(args)