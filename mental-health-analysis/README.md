# Mental Health Analysis Project

This project implements a machine learning pipeline for analyzing mental health data, with a focus on depression prediction. The project uses a modular architecture that allows for easy customization of models, data preprocessing, and feature engineering components.

## Project Structure

```
mental-health-analysis/
├── data/                      # Data directory
│   ├── train.csv             # Training dataset
│   └── test.csv              # Test dataset
├── src/                      # Source code
│   ├── data_analysis/        # Model building and analysis
│   ├── data_cleaning/        # Data cleaning utilities
│   ├── data_encoding/        # Feature encoding
│   ├── data_preprocessing/   # Data preprocessing
│   ├── feature_selection/    # Feature selection
│   └── main.py              # Main entry point
└── notebooks/               # Jupyter notebooks for analysis
    ├── mental_health_analysis.ipynb
    ├── mental_health_eda.ipynb
    └── depression-analysis.ipynb
```

## Installation

1. Clone the repository
2. Install the required dependencies:
```bash
pip3 install -r requirements.txt
```

## Usage

### Running the Analysis

To run the analysis pipeline with default settings:

```bash
python3 src/main.py
```

### Customizing Components

#### Models

The project uses a `ModelBuilder` class that supports various regression models. To change or customize the model:

```python
from src.data_analysis.model_builder import ModelBuilder

# Initialize the model builder
model = ModelBuilder()

# Set the model type (available options: 'linear', 'ridge', 'lasso', 'decision_tree', 'random_forest', 'gradient_boosting')
model.set_model('random_forest', random_state=42)

# Configure hyperparameters for tuning
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
model.set_hyperparameters(params)

# Train and tune the model
model.tune(X_train, y_train)
model.evaluate(X_test, y_test)
```

#### Data Preprocessing

To customize the data preprocessing pipeline:

1. Navigate to `src/data_preprocessing/`
2. Modify or extend the preprocessing steps
3. Update the preprocessing configuration in `main.py`

#### Feature Encoding

The project supports various encoding methods for categorical variables. To customize:

1. Navigate to `src/data_encoding/`
2. Implement your custom encoding logic
3. Update the encoding configuration in `main.py`

#### Data Cleaning

To modify data cleaning procedures:

1. Navigate to `src/data_cleaning/`
2. Customize the cleaning logic
3. Update the cleaning configuration in `main.py`

## Model Evaluation

The `ModelBuilder` class provides built-in evaluation metrics:

```python
# Get evaluation metrics
metrics = model.get_evaluation_metrics()
print(f"MSE: {metrics['mse']}")
print(f"R2 Score: {metrics['r2']}")

# Get best hyperparameters after tuning
best_params = model.get_best_params()
print(f"Best parameters: {best_params}")
```

## Jupyter Notebooks

The project includes several Jupyter notebooks for interactive analysis:

- `mental_health_analysis.ipynb`: Main analysis notebook
- `mental_health_eda.ipynb`: Exploratory Data Analysis
- `depression-analysis.ipynb`: Depression-specific analysis
