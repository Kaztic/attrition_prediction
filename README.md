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

This project predicts employee attrition using machine learning techniques. It includes a data generation pipeline, preprocessing steps, and a Streamlit frontend for visualization and interaction.

## Prerequisites

- Python 3.9 or higher
- Conda (for environment management)
- Git (for cloning the repository)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Kaztic/attrition_prediction.git
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

The application follows a specific workflow to generate data, preprocess it, train the model, and finally run the Streamlit interface. Here's how to run it:

1. **Generate the dataset:**

   ```bash
   python src/data_generator.py
   ```

   This will create a synthetic dataset in the `data/` directory.

2. **Preprocess the data:**

   ```bash
   python src/data_preprocessor.py
   ```

   This will process the generated data and prepare it for model training.

3. **Train the model:**

   ```bash
   python src/model.py
   ```

   This will train the model and save it in the `models/` directory.

4. **Run the Streamlit application:**

   ```bash
   streamlit run app/app.py
   ```

   This will launch the Streamlit interface where you can interact with the model.

## Project Structure

- `src/`: Contains the source code
  - `data_generator.py`: Generates synthetic employee data
  - `data_preprocessor.py`: Preprocesses the generated data
  - `model.py`: Trains and saves the prediction model
- `app/`: Contains the Streamlit application code
  - `app.py`: Main Streamlit application
  - `utils.py`: Utility functions for the app
- `data/`: Directory for storing generated and processed data
- `models/`: Directory for storing trained models
- `requirements.txt`: Lists all the Python packages required for the project

## Features

- Synthetic data generation for employee records
- Data preprocessing and feature engineering
- Machine learning model for attrition prediction
- Interactive Streamlit interface for predictions
- Visualization of model results and insights

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.