from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class DataEncoder:
    def __init__(self):
        self.encoders = {}
        self.encoding_info = {}

    def add_encoding(self, column: str, encoding_type: str, **kwargs) -> 'DataEncoder':
        """Add encoding configuration for a column

        Args:
            column: Name of the column to encode
            encoding_type: Type of encoding ('label', 'onehot', 'ordinal')
            **kwargs: Additional arguments for the encoder
        """
        self.encoding_info[column] = {
            'type': encoding_type,
            'params': kwargs
        }
        return self

    def fit(self, df: pd.DataFrame) -> 'DataEncoder':
        """Fit encoders on the data"""
        for column, info in self.encoding_info.items():
            if info['type'] == 'label':
                encoder = LabelEncoder()
                encoder.fit(df[column].astype(str))
                self.encoders[column] = encoder
            elif info['type'] == 'onehot':
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoder.fit(df[[column]])
                self.encoders[column] = encoder
            elif info['type'] == 'ordinal':
                if 'categories' not in info['params']:
                    raise ValueError(f"Must provide categories for ordinal encoding of {column}")
                categories = info['params']['categories']
                mapping = {cat: i for i, cat in enumerate(categories)}
                self.encoders[column] = mapping
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using fitted encoders"""
        df_encoded = df.copy()
        
        for column, info in self.encoding_info.items():
            encoder = self.encoders[column]
            
            if info['type'] == 'label':
                df_encoded[column] = encoder.transform(df[column].astype(str))
            
            elif info['type'] == 'onehot':
                encoded_cols = encoder.transform(df[[column]])
                feature_names = [f"{column}_{cat}" for cat in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded_cols, columns=feature_names, index=df.index)
                df_encoded = df_encoded.drop(columns=[column]).join(encoded_df)
            
            elif info['type'] == 'ordinal':
                df_encoded[column] = df[column].map(encoder)
        
        return df_encoded

    def fit_transform(self, df):
        """Fit and transform the data"""
        # Skip encoding if input is numpy array (already preprocessed)
        if isinstance(df, np.ndarray):
            return df
            
        return self.fit(df).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform encoded data back to original format"""
        df_decoded = df.copy()
        
        for column, info in self.encoding_info.items():
            encoder = self.encoders[column]
            
            if info['type'] == 'label':
                if column in df_decoded.columns:
                    df_decoded[column] = encoder.inverse_transform(df_decoded[column])
            
            elif info['type'] == 'onehot':
                encoded_cols = [col for col in df_decoded.columns if col.startswith(f"{column}_")]
                if encoded_cols:
                    encoded_data = df_decoded[encoded_cols].values
                    original_data = encoder.inverse_transform(encoded_data)
                    df_decoded = df_decoded.drop(columns=encoded_cols)
                    df_decoded[column] = original_data.ravel()
            
            elif info['type'] == 'ordinal':
                if column in df_decoded.columns:
                    reverse_mapping = {v: k for k, v in encoder.items()}
                    df_decoded[column] = df_decoded[column].map(reverse_mapping)
        
        return df_decoded