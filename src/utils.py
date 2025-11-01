import os
import sys
import pandas as pd
import numpy as np
import pickle
import dill
from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):
    try:
        with open(file_path,'wb') as file:
            pickle.dump(obj,file)
            logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e,sys)