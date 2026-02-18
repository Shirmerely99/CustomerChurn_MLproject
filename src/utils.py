import os
import sys

import pandas as pd
import numpy as np

from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE 

import pickle

def save_object(file_path,obj):
     try:
          dir_path = os.path.dirname(file_path)
          os.makedirs(dir_path, exist_ok=True)
          with open(file_path,"wb") as myfile:
               pickle.dump(obj,myfile)

     except Exception as e:
          raise CustomException(e,sys)
     
def evaluate_models(X_train,y_train,X_test,y_test,models,param):
     try:
          report={}

          for i in range(len(list(models))):
               model = list(models.values())[i]
               params = param[list(models.keys())[i]]

               gs = GridSearchCV(model, params, cv=3)
               gs.fit(X_train,y_train)

               model.set_params(**gs.best_params_)
               model.fit(X_train,y_train)

               preds = model.predict(X_test)
               acc = accuracy_score(preds, y_test)

               report[list(models.keys())[i]]= acc

          return report
     
     except Exception as e:
          raise CustomException(e,sys)


def load_object(file_path):
     try:
          with open(file_path, 'rb') as myfile:
               return pickle.load(myfile)
          
     except Exception as e:
          raise CustomException(e,sys)
