from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

class DataPreprocessor:
    def __init__(self):
        self.num_transformer = None
        self.cat_transformer = None
        self.preprocessor = None
        self.num_cols = []
        self.cat_cols = []

    def set_numerical_columns(self, columns):
        """Set numerical columns for preprocessing"""
        self.num_cols = columns
        return self

    def set_categorical_columns(self, columns):
        """Set categorical columns for preprocessing"""
        self.cat_cols = columns
        return self

    def build_numerical_transformer(self, imputer_strategy="median", use_scaler=True):
        """Build numerical transformer pipeline"""
        steps = [('imputer', SimpleImputer(strategy=imputer_strategy))]
        if use_scaler:
            steps.append(('scaler', StandardScaler()))
        self.num_transformer = Pipeline(steps=steps)
        return self

    def build_categorical_transformer(self, imputer_strategy="most_frequent"):
        """Build categorical transformer pipeline"""
        from sklearn.preprocessing import OneHotEncoder
        steps = [
            ('imputer', SimpleImputer(strategy=imputer_strategy)),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
        ]
        self.cat_transformer = Pipeline(steps=steps)
        return self

    def build(self):
        """Build the complete preprocessing pipeline"""
        if not self.num_transformer or not self.cat_transformer:
            raise ValueError("Both numerical and categorical transformers must be built first")

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.num_transformer, self.num_cols),
                ('cat', self.cat_transformer, self.cat_cols)
            ]
        )
        return self.preprocessor

    def fit_transform(self, data):
        """Apply preprocessing transformations"""
        if not self.preprocessor:
            raise ValueError("Preprocessor must be built first")
    
        # Handle conditional columns based on profession
        data['Academic Pressure'] = data['Academic Pressure'].fillna(0)
        data['Work Pressure'] = data['Work Pressure'].fillna(0)
        data['Study Satisfaction'] = data['Study Satisfaction'].fillna(0)
        data['Job Satisfaction'] = data['Job Satisfaction'].fillna(0)
        data['CGPA'] = data['CGPA'].fillna(0)
    
        return self.preprocessor.fit_transform(data)

    def transform(self, X):
        """Transform the data"""
        if not self.preprocessor:
            raise ValueError("Preprocessor must be built first")
        return self.preprocessor.transform(X)