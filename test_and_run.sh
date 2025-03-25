#!/bin/bash

# Start FastAPI server in the background
echo "Starting FastAPI server..."
python3 main.py &
FASTAPI_PID=$!

# Wait for server to start
sleep 5

# Test endpoints
echo "Testing endpoints..."

# Test root endpoint
echo "Testing root endpoint..."
curl -s http://localhost:8000/ | grep -q "name" && echo "✓ Root endpoint test passed" || echo "✗ Root endpoint test failed"

# Test data generation
echo "Testing data generation..."
curl -s -X POST http://localhost:8000/generate-data -H "Content-Type: application/json" -d '{"n_employees": 100}' | grep -q "message" && echo "✓ Data generation test passed" || echo "✗ Data generation test failed"

# Test data processing
echo "Testing data processing..."
curl -s -X POST http://localhost:8000/process-data | grep -q "message" && echo "✓ Data processing test passed" || echo "✗ Data processing test failed"

# Test model training
echo "Testing model training..."
curl -s -X POST http://localhost:8000/train-model | grep -q "message" && echo "✓ Model training test passed" || echo "✗ Model training test failed"

# Test prediction
echo "Testing prediction..."
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"employee_data": {"age": 35, "department": "Sales", "distance_from_home": 10, "education": 3, "environment_satisfaction": 4, "job_involvement": 3, "job_level": 2, "job_satisfaction": 4, "monthly_income": 5000, "num_companies_worked": 2, "percent_salary_hike": 15, "performance_rating": 3, "relationship_satisfaction": 4, "stock_option_level": 1, "total_working_years": 8, "training_times_last_year": 2, "work_life_balance": 3, "years_at_company": 5, "years_in_current_role": 3, "years_since_last_promotion": 2, "years_with_curr_manager": 3}}' | grep -q "prediction" && echo "✓ Prediction test passed" || echo "✗ Prediction test failed"

# Kill FastAPI server
kill $FASTAPI_PID

# Start Streamlit dashboard
echo "Starting Streamlit dashboard..."
streamlit run app/app.py 