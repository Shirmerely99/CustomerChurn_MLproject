import os
import sys
from src.logger import logging
from src.exception import CustomException

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from dataclasses import dataclass

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    model_path = os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__ (self):
        self.model_trainer= ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        logging.info(f"Model trainer initiated")
        try:
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            logging.info(f"Train and Test dataset spilt into X and y")

            models={
                'Logistic Regression': LogisticRegression(),
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'SVC': SVC(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'KNN': KNeighborsClassifier()
                    }
            
            logging.info(f"Hyperparameter tuning initiated")
            
            params = {
                'Logistic Regression': {
                    'C': [0.1, 1, 10]
                    },
                'Decision Tree': {
                    'max_depth': [3, 5, 10],
                    'criterion': ['gini', 'entropy']
                    },
                'Random Forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 5, 10]
                    },
                'SVC': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear']
                    },
                'KNN': {
                    'n_neighbors': [3, 5, 11],
                    'weights': ['uniform', 'distance']
                    },
                'Gradient Boosting': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1]
                    },
                    }
    
            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
            model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(model_score)]
            best_model= models[best_model_name]

            if model_score<0.65:
                raise CustomException("No best model.")
            
            best_model.fit(X_train, y_train)

            save_object(file_path=self.model_trainer.model_path,
                        obj=best_model)
            logging.info(f"Best model found AND model training completed")
            
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_pred, y_test)
            
            return accuracy

        except Exception as e:
            raise CustomException(e,sys)