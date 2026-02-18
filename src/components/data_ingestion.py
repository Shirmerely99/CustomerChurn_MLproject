import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig 

@dataclass
class DataIngestionConfig:
    train_path:str = os.path.join('artifacts',"train.csv")
    test_path:str = os.path.join('artifacts',"test.csv")
    raw_path: str = os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__ (self):
        self.data_ingestion = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion initiated")
        try:
            df = pd.read_csv('data/Bank_Customer_Churn_Engineered.csv')
            logging.info("Dataset read as dataframe")

            os.makedirs(os.path.dirname(self.data_ingestion.train_path), exist_ok=True)

            df.to_csv(self.data_ingestion.raw_path, index=False, header=True)

            logging.info("Train and Test split initiated")
            train_data, test_data= train_test_split(df, test_size=0.2, random_state=32)

            train_data.to_csv(self.data_ingestion.train_path, index=False, header=True)
            test_data.to_csv(self.data_ingestion.test_path, index=False, header=True)
            logging.info("Train and Test data saved as csv files")

            logging.info("Data ingestion completed")

            return (self.data_ingestion.train_path,
                    self.data_ingestion.test_path)
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
if __name__ =="__main__":
    inges_obj= DataIngestion()
    train_data,test_data = inges_obj.initiate_data_ingestion()

    trans_obj= DataTransformation()
    train_arr,test_arr,_ = trans_obj.initiate_data_transformation(train_data, test_data)

    trainer_obj= ModelTrainer()
    print(trainer_obj.initiate_model_trainer(train_arr,test_arr))