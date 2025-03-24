from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from typing import Optional, List
from pydantic import BaseModel
import uvicorn

from src.data_generator import generate_synthetic_data
from src.feature_engineering import engineer_features
from src.model import build_model, save_model, evaluate_model, load_model, predict_attrition

app = FastAPI(
    title="Employee Attrition Prediction API",
    description="API for predicting employee attrition using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmployeeData(BaseModel):
    n_employees: int = 1000

class PredictionRequest(BaseModel):
    employee_data: dict

@app.post("/generate-data")
async def generate_data(request: EmployeeData):
    """Generate synthetic employee data"""
    try:
        df = generate_synthetic_data(n_employees=request.n_employees)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/employee_data.csv", index=False)
        return {"message": f"Generated {request.n_employees} employee records", "file": "data/employee_data.csv"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-data")
async def process_data():
    """Process and engineer features from the employee data"""
    try:
        df = pd.read_csv("data/employee_data.csv")
        df = engineer_features(df)
        df.to_csv("data/processed_employee_data.csv", index=False)
        return {"message": "Data processed successfully", "file": "data/processed_employee_data.csv"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Employee data not found. Generate data first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train-model")
async def train_model():
    """Train and save the model"""
    try:
        df = pd.read_csv("data/processed_employee_data.csv")
        model, explainer, X_test, y_test, features = build_model(df)
        os.makedirs("models", exist_ok=True)
        save_model(model, explainer, features)
        
        # Get evaluation metrics
        evaluation = evaluate_model(model, X_test, y_test)
        return {"message": "Model trained and saved successfully", "evaluation": evaluation}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Processed data not found. Process data first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Predict attrition for a single employee"""
    try:
        # Convert input dictionary to DataFrame
        employee_df = pd.DataFrame([request.employee_data])
        
        # Load model and make prediction
        model, explainer, features = load_model()
        predictions, probabilities, explanations = predict_attrition(employee_df, model, explainer, features)
        
        # Get prediction and probability
        prediction = bool(predictions[0])
        probability = float(probabilities[0])
        
        # Get feature importance from SHAP values if available
        explanation = []
        if explanations is not None:
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': abs(explanations[0])
            }).sort_values('importance', ascending=False)
            
            # Convert feature importance to list of dicts for JSON response
            explanation = feature_importance.head(10).to_dict('records')
        
        return {
            "prediction": prediction,
            "probability": probability,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Employee Attrition Prediction API",
        "version": "1.0.0",
        "endpoints": [
            "/generate-data",
            "/process-data",
            "/train-model",
            "/predict"
        ]
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)