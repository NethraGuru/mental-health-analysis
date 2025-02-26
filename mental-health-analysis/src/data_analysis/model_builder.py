import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from typing import Dict, Any, Optional, List
import pandas as pd

class ModelBuilder:
    def __init__(self):
        self.model = None
        self.params = None
        self.is_tuned = False
        self.best_params = None
        self.evaluation_metrics = {}

    def set_model(self, model_type: str, **kwargs) -> 'ModelBuilder':
        """Set the model type and its parameters"""
        models = {
            'linear': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'decision_tree': DecisionTreeRegressor,
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor
        }

        if model_type not in models:
            raise ValueError(f"Model type {model_type} not supported")

        self.model = models[model_type](**kwargs)
        return self

    def set_hyperparameters(self, params: Dict[str, Any]) -> 'ModelBuilder':
        """Set hyperparameters for grid search"""
        self.params = params
        return self

    def tune(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> 'ModelBuilder':
        """Perform hyperparameter tuning using grid search"""
        if self.params is None:
            raise ValueError("Hyperparameters must be set before tuning")

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.params,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.is_tuned = True
        return self

    def train(self, X: pd.DataFrame, y: pd.Series) -> 'ModelBuilder':
        """Train the model"""
        if self.model is None:
            raise ValueError("Model must be set before training")

        self.model.fit(X, y)
        return self

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate the model performance"""
        if not hasattr(self.model, 'predict'):
            raise ValueError("Model must be trained before evaluation")

        y_pred = self.model.predict(X)
        self.evaluation_metrics = {
            'mse': mean_squared_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        return self.evaluation_metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model"""
        if not hasattr(self.model, 'predict'):
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(X)

    def get_model(self):
        """Get the trained model"""
        return self.model

    def get_best_params(self) -> Optional[Dict]:
        """Get the best parameters from tuning"""
        return self.best_params if self.is_tuned else None

    def get_evaluation_metrics(self) -> Dict[str, float]:
        """Get the evaluation metrics"""
        return self.evaluation_metrics