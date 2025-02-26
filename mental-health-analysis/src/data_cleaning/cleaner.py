import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional

class DataCleaner:
    def __init__(self):
        self.outlier_threshold = 1.5
        self.missing_threshold = 0.5

    def set_outlier_threshold(self, threshold: float) -> 'DataCleaner':
        """Set the threshold for outlier detection (IQR method)"""
        self.outlier_threshold = threshold
        return self

    def set_missing_threshold(self, threshold: float) -> 'DataCleaner':
        """Set the threshold for dropping columns with too many missing values"""
        self.missing_threshold = threshold
        return self

    def remove_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Remove outliers using the IQR method"""
        df_clean = df.copy()
        for column in columns:
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.outlier_threshold * IQR
            upper_bound = Q3 + self.outlier_threshold * IQR
            df_clean = df_clean[(df_clean[column] >= lower_bound) & 
                               (df_clean[column] <= upper_bound)]
        return df_clean

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values by dropping columns with too many missing values"""
        missing_ratio = df.isnull().sum() / len(df)
        columns_to_drop = missing_ratio[missing_ratio > self.missing_threshold].index
        return df.drop(columns=columns_to_drop)

    def validate_data(self, df: pd.DataFrame, rules: Dict[str, Dict]) -> pd.DataFrame:
        """Validate data based on provided rules
        
        rules format:
        {
            'column_name': {
                'type': 'numeric'/'categorical',
                'range': (min, max),  # for numeric
                'categories': [...]    # for categorical
            }
        }
        """
        df_validated = df.copy()
        
        for column, rule in rules.items():
            if rule['type'] == 'numeric':
                if 'range' in rule:
                    min_val, max_val = rule['range']
                    df_validated = df_validated[
                        (df_validated[column] >= min_val) & 
                        (df_validated[column] <= max_val)
                    ]
            elif rule['type'] == 'categorical':
                if 'categories' in rule:
                    df_validated = df_validated[
                        df_validated[column].isin(rule['categories'])
                    ]
        
        return df_validated

    def clean(self, df: Union[pd.DataFrame, np.ndarray], 
              numeric_columns: List[str], 
              validation_rules: Optional[Dict] = None) -> Union[pd.DataFrame, np.ndarray]:
        """Apply all cleaning steps in sequence"""
        # If input is numpy array, skip cleaning as it's already preprocessed
        if isinstance(df, np.ndarray):
            return df
            
        # For DataFrame, proceed with normal cleaning
        existing_numeric_cols = [col for col in numeric_columns if col in df.columns]
        
        df_cleaned = self.handle_missing_values(df)
        df_cleaned = self.remove_outliers(df_cleaned, existing_numeric_cols)
        
        if validation_rules:
            df_cleaned = self.validate_data(df_cleaned, validation_rules)
            
        return df_cleaned