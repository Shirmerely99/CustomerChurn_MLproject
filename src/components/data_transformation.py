import os 
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC

from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    transformer_path = os.path.join('artifacts',"transformer.pkl")

class DataTransformation:
    def __init__ (self):
        self.data_transformation = DataTransformationConfig()

    def get_data_transformer(self):
        try:
            num_columns = ['age', 'credit_score', 'balance', 'estimated_salary', 'tenure',
                           'BalanceSalaryRatio', 'TenureByAge', 'CreditScoreByAge']
            cat_columns = ['country', 'gender', 'credit_card', 'active_member', 'products_number']

            num_pipeline = Pipeline(
                steps=[('skewed', FunctionTransformer(np.log1p, validate=True)),
                       ('outliers', RobustScaler()),
                       ("Scaler", StandardScaler())])
            
            cat_pipeline = Pipeline(
                steps=[("one-hot", OneHotEncoder(handle_unknown='ignore', drop='first'))])
            
            transformer = ColumnTransformer(
                [("numerical columns",num_pipeline,num_columns),
                 ("categorial columns",cat_pipeline,cat_columns)])
            
            logging.info(f"Categorical columns: {cat_columns}")
            logging.info(f"Numerical columns: {num_columns}")
            
            return transformer
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self, train_data, test_data):
        logging.info("Data Tranformation initiated")
        try:
            train_data = pd.read_csv(train_data)
            test_data = pd.read_csv(test_data)

            logging.info("training and testing dataset read as dataframe")
            logging.info(f"train_data shape: {train_data.shape}")
            logging.info(f"test_data shape: {test_data.shape}")

            target_column = "churn"
            num_column = ['age', 'credit_score', 'balance', 'estimated_salary', 'tenure',
                           'BalanceSalaryRatio', 'TenureByAge', 'CreditScoreByAge']
            
            X_train = train_data.drop(columns=[target_column])
            y_train = train_data[target_column]

            X_test = test_data.drop(columns=[target_column])
            y_test = test_data[target_column]

            logging.info(f"X_train shape: {X_train.shape}")
            logging.info(f"y_train shape: {y_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}")
            logging.info(f"y_test shape: {y_test.shape}")

            transformer_obj = self.get_data_transformer()
            X_train_arr = transformer_obj.fit_transform(X_train)
            X_test_arr = transformer_obj.transform(X_test)
            logging.info("Applied the transforming object on train and test dataframe.")

            logging.info(f"X_train_arr shape: {X_train_arr.shape}")
            logging.info(f"y_train shape: {y_train.shape}")
            logging.info(f"X_test_arr shape: {X_test_arr.shape}")
            logging.info(f"y_test shape: {y_test.shape}")
            
            # Merge the X_array and y into a complete array
            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            save_object(file_path= self.data_transformation.transformer_path,
                        obj= transformer_obj)
            logging.info("Saved the transforming object.")
            
            logging.info("Data Transformation completed")
            
            return (train_arr,
                    test_arr,
                    self.data_transformation.transformer_path)
        
        except Exception as e:
            raise CustomException(e,sys)