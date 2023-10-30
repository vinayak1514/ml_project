from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import os
import sys 
import pandas as pd 
from src.logger import logging 
from src.exception import CustomException

obj =DataIngestion()
train_data_path,test_data_path = obj.initiate_data_ingestion()
data_transformation = DataTransformation()
train_arr , test_arr = data_transformation.initiate_data_transformation(train_data_path,test_data_path)

model = ModelTrainer()
model.initiate_model_trainer(train_arr,test_arr)