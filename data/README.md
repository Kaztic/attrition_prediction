# AI-Powered Attrition Prediction

This project implements an AI-driven employee attrition prediction system that helps HR teams, managers, and leadership proactively identify and retain at-risk employees.

## Features

- **Predictive Analytics**: Machine learning model to identify employees at risk of leaving
- **Explainable AI**: SHAP-based explanations for attrition risk factors
- **Interactive Dashboard**: Streamlit visualization for exploring attrition patterns
- **Recommendation Engine**: Personalized retention strategies based on risk factors
- **Risk Clustering**: Identification of groups with similar attrition drivers

## Project Structure

```
attrition_prediction/
├── app/                # Streamlit dashboard application
│   ├── __init__.py
│   └── app.py          # Dashboard implementation
├── data/               # Data storage
│   └── README.md
├── models/             # Model storage
│   └── README.md
├── src/                # Source code
│   ├── __init__.py
│   ├── data_generator.py       # Synthetic data creation
│   ├── feature_engineering.py  # Feature creation and preprocessing
│   ├── model.py               # Model training and evaluation
│   └── recommendation_engine.py # Retention recommendations logic
├── environment.yml     # Conda environment configuration
├── main.py             # Main application entry point
└── setup.sh            # Project setup script
```

## Prerequisites

- Python 3.9+
- Conda package manager
- Git

## Setup

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/attrition-prediction.git
   cd attrition-prediction
   ```

2. Run the setup script to create the project structure
   ```bash
   bash setup.sh
   ```

3. Create the conda environment
   ```bash
   conda env create -f environment.yml
   ```

4. Activate the environment
   ```bash
   conda activate attrition-prediction
   ```

## Usage

The application can be run with the following commands:

```bash
# Generate synthetic employee data
python main.py --generate-data --n-employees 1000

# Process data and engineer features
python main.py --process-data

# Train the prediction model
python main.py --train-model

# Launch the dashboard
python main.py --run-dashboard
```

Or run all steps sequentially:

```bash
python main.py --generate-data --process-data --train-model --run-dashboard
```

## Dashboard Features

The Streamlit dashboard provides three main tabs:

### 1. Overview Tab
- Organization-wide attrition risk metrics
- Department and team-level risk visualizations
- Risk distribution by employee demographics
- Interactive filters for customized views

### 2. Employee Risk Analysis
- Detailed individual employee risk profiles
- SHAP visualizations to explain risk factors
- Historical trend analysis
- Peer comparison

### 3. Recommendations
- Risk cluster identification
- Personalized retention strategies
- Prioritized action plans
- General best practices for retention

## Model Details

The prediction model uses a Random Forest classifier with the following characteristics:

- **Algorithm**: Random Forest Classifier
- **Features**: 20+ engineered features including tenure, performance, engagement signals
- **Explainability**: SHAP (SHapley Additive exPlanations) values
- **Evaluation**: Precision, Recall, F1-Score, ROC AUC
- **Bias Mitigation**: Class weighting and fairness considerations

## Data Sources

The system uses the following data types:

- **Employee Data**: Tenure, role history, performance, skill relevance
- **Engagement Signals**: Training participation, survey satisfaction
- **Team Dynamics**: Peer attrition, manager relationship
- **Career Growth**: Promotions, role changes, skill development

For the hackathon, we generate synthetic data that mimics real-world patterns while ensuring privacy.

## Evaluation Metrics

The model's performance is evaluated using:

- **Precision**: Ability to correctly identify true attrition risks
- **Recall**: Ability to capture all potential attrition cases
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the Receiver Operating Characteristic curve
- **Business Impact**: Estimated retention improvement and cost savings

## Extending the Project

To extend this project:

1. **Add Real Data**: Replace synthetic data generator with connections to real HR systems
2. **Enhance Features**: Integrate additional data sources like survey responses or communication patterns
3. **Improve Model**: Experiment with different algorithms or ensemble approaches
4. **Expand Dashboard**: Add additional visualizations or reporting capabilities
5. **Integration**: Connect with existing HR systems for automated workflows

## Contributing

We welcome contributions to improve the attrition prediction system:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Submit a pull request

Please follow coding standards and include tests for new functionality.

## License

This project is licensed for internal use only. All rights reserved.

## Acknowledgments

- SHAP library for model explanations
- Streamlit for interactive visualization
- Scikit-learn for machine learning algorithms