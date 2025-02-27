import pandas as pd
from src.data_preprocessing.preprocessor import DataPreprocessor
from src.data_cleaning.cleaner import DataCleaner
from src.data_encoding.encoder import DataEncoder
from src.feature_selection.selector import FeatureSelector
from src.data_analysis.model_builder import ModelBuilder

class MentalHealthAnalysisPipeline:
    def __init__(self):
        self.preprocessor = None
        self.cleaner = None
        self.encoder = None
        self.feature_selector = None
        self.model_builder = None
        self.data = None
        self.target = None

    def load_data(self, train_path: str, test_path: str = None):
        """Load the dataset(s)"""
        self.data = pd.read_csv(train_path)
        if test_path:
            self.test_data = pd.read_csv(test_path)
        return self

    def set_target(self, target_column: str):
        """Set the target variable"""
        if target_column not in self.data.columns:
            raise ValueError(f"Target column {target_column} not found in data")
        self.target = self.data[target_column]
        self.data = self.data.drop(columns=[target_column])
        return self

    def configure_preprocessing(self, num_cols: list, cat_cols: list):
        """Configure the preprocessing step"""
        self.preprocessor = DataPreprocessor()
        self.preprocessor.set_numerical_columns(num_cols)\
                        .set_categorical_columns(cat_cols)\
                        .build_numerical_transformer()\
                        .build_categorical_transformer()\
                        .build()
        return self

    def configure_cleaning(self, outlier_threshold: float = 1.5,
                         missing_threshold: float = 0.5):
        """Configure the data cleaning step"""
        self.cleaner = DataCleaner()
        self.cleaner.set_outlier_threshold(outlier_threshold)\
                    .set_missing_threshold(missing_threshold)
        return self

    def configure_encoding(self, encoding_config: dict):
        """Configure the encoding step"""
        self.encoder = DataEncoder()
        for column, config in encoding_config.items():
            self.encoder.add_encoding(column, **config)
        return self

    def configure_feature_selection(self, method: str, **kwargs):
        """Configure the feature selection step"""
        self.feature_selector = FeatureSelector()
        if method == 'correlation':
            self.feature_selector.using_correlation(**kwargs)
        elif method == 'mutual_info':
            self.feature_selector.using_mutual_info(**kwargs)
        elif method == 'rfe':
            self.feature_selector.using_recursive_elimination(**kwargs)
        return self

    def configure_model(self, model_type: str, params: dict = None):
        """Configure the model"""
        self.model_builder = ModelBuilder()
        self.model_builder.set_model(model_type)
        if params:
            self.model_builder.set_hyperparameters(params)
        return self

    def run(self):
        """Execute the analysis pipeline"""
        if self.data is None:
            raise ValueError("Data must be loaded first")
    
        # Preprocess the data first
        if self.preprocessor:
            self.data = self.preprocessor.fit_transform(self.data)
    
        # Clean the data after preprocessing
        if self.cleaner:
            self.data = self.cleaner.clean(self.data, 
                                         self.preprocessor.num_cols)
    
        # Encode categorical variables
        if self.encoder:
            self.data = self.encoder.fit_transform(self.data)
    
        # Select features
        if self.feature_selector:
            self.data = self.feature_selector.fit_transform(self.data, self.target)
    
        # Train and evaluate model
        if self.model_builder:
            self.model_builder.train(self.data, self.target)
            metrics = self.model_builder.evaluate(self.data, self.target)
            print("Model Performance:")
            print(f"MSE: {metrics['mse']:.4f}")
            print(f"R2 Score: {metrics['r2']:.4f}")
    
        return self

def main():
    pipeline = MentalHealthAnalysisPipeline()
    
    # Load and configure data
    pipeline.load_data('data/train.csv', 'data/test.csv')\
            .set_target('Depression')

    # Get available columns from the dataset
    available_cols = pipeline.data.columns

    # Configure preprocessing with only available columns
    all_num_cols = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction',
                'Job Satisfaction', 'Work/Study Hours', 'Financial Stress']
    all_cat_cols = ['Gender', 'City', 'Working Professional or Student', 'Profession',
                'Sleep Duration', 'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?',
                'Family History of Mental Illness']
    
    # Filter columns that exist in the dataset
    num_cols = [col for col in all_num_cols if col in available_cols]
    cat_cols = [col for col in all_cat_cols if col in available_cols]
    
    pipeline.configure_preprocessing(num_cols, cat_cols)

    # Configure cleaning
    pipeline.configure_cleaning(outlier_threshold=1.5)

    # Configure encoding
    encoding_config = {
        'Gender': {'encoding_type': 'label'},
        'City': {'encoding_type': 'onehot'},
        'Profession': {'encoding_type': 'onehot'}
    }
    pipeline.configure_encoding(encoding_config)

    # Configure feature selection
    pipeline.configure_feature_selection('mutual_info', k=10)

    # Configure and run model
    model_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None]
    }
    pipeline.configure_model('random_forest', params=model_params)\
            .run()

if __name__ == '__main__':
    main()
