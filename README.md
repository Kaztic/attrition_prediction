---
title: Attrition Prediction
emoji: 😻
colorFrom: blue
colorTo: pink
sdk: streamlit
sdk_version: 1.44.0
app_file: app.py
pinned: false
short_description: Predict the attrition like a pro..!
---

# Employee Attrition Prediction

This project predicts employee attrition using machine learning techniques. It includes a FastAPI backend for data processing and a Streamlit frontend for visualization and interaction.

## Prerequisites

- Python 3.9 or higher
- Conda (for environment management)
- Git (for cloning the repository)

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd attrition-prediction
   ```

2. **Create a new Conda environment:**

   ```bash
   conda create --name attrition-prediction python=3.9
   ```

3. **Activate the Conda environment:**

   ```bash
   conda activate attrition-prediction
   ```

4. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Option 1: Using the Run Script (Recommended)

The easiest way to run the application is to use the provided script:

1. **Make the script executable:**

   ```bash
   chmod +x test_and_run.sh
   ```

2. **Run the script:**

   ```bash
   ./test_and_run.sh
   ```

This script will:
- Start the FastAPI server
- Launch the Streamlit application
- Handle proper shutdown of both services when you're done

### Option 2: Manual Startup

If you prefer to run the services manually:

1. **Start the FastAPI server:**

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Launch the Streamlit app (in a new terminal):**

   ```bash
   streamlit run app/app.py
   ```

3. **Access the application:**

   Open your web browser and navigate to `http://localhost:8501` to access the Streamlit app.

## Project Structure

- `src/`: Contains the source code for data generation, feature engineering, and model building
- `app/`: Contains the Streamlit application code
- `data/`: Directory for storing generated and processed data
- `models/`: Directory for storing trained models
- `requirements.txt`: Lists all the Python packages required for the project
- `test_and_run.sh`: Script to start both the FastAPI server and Streamlit app

## API Features

- Data generation and processing
- Model training and evaluation
- Real-time attrition predictions
- Manager and employee dialogue generation using Gemini AI
- Retention recommendations

## Contributing

Feel free to open issues or submit pull requests for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.