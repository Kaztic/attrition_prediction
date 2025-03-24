#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check and install required dependencies
echo -e "${BLUE}Checking dependencies...${NC}"

# Check for conda
if ! command_exists conda; then
    echo -e "${RED}Conda is not installed. Please install Miniconda or Anaconda first.${NC}"
    exit 1
fi

# Check for streamlit
if ! command_exists streamlit; then
    echo -e "${YELLOW}Streamlit not found. Installing...${NC}"
    conda install -c conda-forge streamlit -y
fi

# Check for curl
if ! command_exists curl; then
    echo -e "${YELLOW}Curl not found. Installing...${NC}"
    conda install -c conda-forge curl -y
fi

echo -e "${BLUE}Starting API endpoint tests...${NC}"

# Function to test an endpoint
test_endpoint() {
    local endpoint=$1
    local method=$2
    local data=$3
    local description=$4
    
    echo -e "\n${BLUE}Testing $description...${NC}"
    
    if [ "$method" = "POST" ]; then
        response=$(curl -s -X POST "http://localhost:8000$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data")
    else
        response=$(curl -s "http://localhost:8000$endpoint")
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Success${NC}"
        echo "Response: $response"
    else
        echo -e "${RED}✗ Failed${NC}"
    fi
    
    # Wait for 2 seconds between requests
    sleep 2
}

# Start the FastAPI server in the background
echo -e "${BLUE}Starting FastAPI server...${NC}"
python main.py &
FASTAPI_PID=$!

# Wait for the server to start
echo "Waiting for server to start..."
sleep 5

# Test the root endpoint
test_endpoint "/" "GET" "" "Root endpoint"

# Test generate-data endpoint
test_endpoint "/generate-data" "POST" '{"n_employees": 1000}' "Generate data endpoint"

# Test process-data endpoint
test_endpoint "/process-data" "POST" "" "Process data endpoint"

# Test train-model endpoint
test_endpoint "/train-model" "POST" "" "Train model endpoint"

# Test predict endpoint with sample data
test_endpoint "/predict" "POST" '{
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
}' "Predict endpoint"

# Kill the FastAPI server
echo -e "\n${BLUE}Stopping FastAPI server...${NC}"
kill $FASTAPI_PID

# Wait for the server to stop
sleep 2

# Launch Streamlit app
echo -e "\n${BLUE}Launching Streamlit app...${NC}"
streamlit run app/app.py 