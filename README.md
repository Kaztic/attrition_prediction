# AI-Powered Employee Attrition Prediction System

A comprehensive system for predicting employee attrition using machine learning, featuring both a FastAPI backend and a Streamlit dashboard.

## Features

- Synthetic data generation for testing and development
- Feature engineering and preprocessing
- Machine learning model training with SHAP explanations
- FastAPI backend with RESTful endpoints
- Interactive Streamlit dashboard for visualization
- Automated testing script for all endpoints

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd attrition-prediction
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate attrition-prediction
```

## Usage

### FastAPI Backend

The system provides the following RESTful endpoints:

1. **Root Endpoint**
   - URL: `GET /`
   - Description: Returns API information and available endpoints

2. **Generate Data**
   - URL: `POST /generate-data`
   - Description: Generates synthetic employee data
   - Request body:
     ```json
     {
         "n_employees": 1000
     }
     ```

3. **Process Data**
   - URL: `POST /process-data`
   - Description: Processes and engineers features from the employee data

4. **Train Model**
   - URL: `POST /train-model`
   - Description: Trains and saves the attrition prediction model

5. **Predict**
   - URL: `POST /predict`
   - Description: Makes predictions for individual employees
   - Request body:
     ```json
     {
         "employee_data": {
             "age": 35,
             "department": "Sales",
             "distance_from_home": 10,
             "education": 3,
             "environment_satisfaction": 4,
             "job_involvement": 3,
             "job_level": 2,
             "job_satisfaction": 4,
             "monthly_income": 5000,
             "num_companies_worked": 2,
             "percent_salary_hike": 15,
             "performance_rating": 3,
             "relationship_satisfaction": 4,
             "stock_option_level": 1,
             "total_working_years": 8,
             "training_times_last_year": 2,
             "work_life_balance": 3,
             "years_at_company": 5,
             "years_in_current_role": 3,
             "years_since_last_promotion": 2,
             "years_with_curr_manager": 3
         }
     }
     ```

### Automated Testing

The repository includes a test script that automatically tests all endpoints and launches the Streamlit dashboard:

```bash
./test_and_run.sh
```

This script will:
1. Check and install required dependencies
2. Start the FastAPI server
3. Test all endpoints in sequence
4. Stop the FastAPI server
5. Launch the Streamlit dashboard

### Streamlit Dashboard

The dashboard provides:
- Overview of attrition risk across the organization
- Detailed employee analysis with risk factors
- Retention recommendations based on risk patterns
- Interactive visualizations and filters

To run the dashboard directly:
```bash
streamlit run app/app.py
```

## Project Structure

```
attrition_prediction/
├── app/
│   ├── __init__.py
│   └── app.py              # Streamlit dashboard
├── data/                   # Data storage
├── models/                 # Saved model artifacts
├── src/
│   ├── data_generator.py   # Synthetic data generation
│   ├── feature_engineering.py  # Feature processing
│   ├── model.py           # Model training and prediction
│   └── recommendation_engine.py  # Risk analysis
├── environment.yml        # Conda environment
├── main.py               # FastAPI application
└── test_and_run.sh       # Test script
```

## Dependencies

- Python 3.9
- FastAPI
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- SHAP
- Plotly
- Other dependencies listed in environment.yml

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.