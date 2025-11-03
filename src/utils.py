import os
import sys
import pandas as pd
import numpy as np
import pickle
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):
    try:
        with open(file_path,'wb') as file:
            pickle.dump(obj,file)
            logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}
        
        for i in range(len(models)):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]
            
            gs = GridSearchCV(model,para,cv=5,n_jobs=-1,verbose=1,refit=True)
            gs.fit(X_train,y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            
            
            # Train the model
            #model.fit(X_train,y_train)
            
            # Predicting the test data
            y_test_pred = model.predict(X_test)
            
            # Get the r2 score for the model
            test_model_score = r2_score(y_test,y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)