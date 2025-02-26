import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.linear_model import LinearRegression
from typing import Optional

class FeatureSelector:
    def __init__(self):
        self.selector = None
        self.selected_features = None
        self.method = None

    def using_correlation(self, threshold: float = 0.5):
        """Set up correlation-based feature selection"""
        self.method = 'correlation'
        self.threshold = threshold
        return self

    def using_mutual_info(self, k: int = 10):
        """Set up mutual information based feature selection"""
        self.method = 'mutual_info'
        self.selector = SelectKBest(score_func=mutual_info_regression, k=k)
        return self

    def using_recursive_elimination(self, n_features: int = 10):
        """Configure recursive feature elimination"""
        self.method = 'rfe'
        self.selector = RFE(estimator=LinearRegression(), n_features_to_select=n_features)
        return self

    def fit(self, X, y):
        """Fit the feature selector"""
        if self.method is None:
            raise ValueError("Feature selection method must be configured first")

        # Skip feature selection if input is numpy array (already preprocessed)
        if isinstance(X, np.ndarray):
            return self

        if self.method == 'correlation':
            corr = X.corr()
            self.selected_features = [col for col in X.columns 
                                    if abs(corr[col].mean()) >= self.threshold]
        else:
            self.selector.fit(X, y)
            if self.method == 'mutual_info':
                self.selected_features = X.columns[self.selector.get_support()].tolist()
            elif self.method == 'rfe':
                self.selected_features = X.columns[self.selector.support_].tolist()
        
        return self

    def transform(self, X):
        """Transform the data using selected features"""
        # Skip feature selection if input is numpy array (already preprocessed)
        if isinstance(X, np.ndarray):
            return X

        if self.method == 'correlation':
            return X[self.selected_features]
        return self.selector.transform(X)

    def fit_transform(self, X, y):
        """Fit and transform the data"""
        # Skip feature selection if input is numpy array (already preprocessed)
        if isinstance(X, np.ndarray):
            return X
            
        return self.fit(X, y).transform(X)