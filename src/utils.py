import os
import sys
import pandas as pd
import numpy as np
import pickle
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):
    try:
        with open(file_path,'wb') as file:
            pickle.dump(obj,file)
            logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train the model
            model.fit(X_train,y_train)
            
            # Predicting the test data
            y_test_pred = model.predict(X_test)
            
            # Get the r2 score for the model
            test_model_score = r2_score(y_test,y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
    except Exception as e:
        raise CustomException(e,sys)