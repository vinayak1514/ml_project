import os 
import sys 
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.logger import logging 
from src.exception import CustomException
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from dataclasses import dataclass
from src.utils import save_object,evaluate_models
@dataclass 
class ModelTrainerConfig:
     trained_model_file_path = os.path.join('artifacts','model.pkl')
class ModelTrainer:
    def __init__(self):
       self.model_trainer_config= ModelTrainerConfig()
    
    def  initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test = [
                    train_array[:,:-1],
                    train_array[:,-1],
                    test_array[:,:-1],
                    test_array[:,-1]
            ]
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report:dict = evaluate_models(X_train,X_test,y_train,y_test,models)
            print(model_report)
            logging.info(f'Model Report : {model_report}')
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            logging.info(f'Best Model Found-Model Name : {best_model_name} , R2 Score : {best_model_score}')
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
        except Exception as e:
            logging.info('error accured')
            raise CustomException(e,sys)

