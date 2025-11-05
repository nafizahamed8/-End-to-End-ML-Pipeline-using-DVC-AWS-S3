import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier

#ensuring
log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

#setting up logger
logger=logging.getLogger("model_training")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"model_training.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str):
    """load data from csv file
    :param file_path:path to the csv file
    :return loaded dataframe"""

    try:
        df=pd.read_csv(file_path)
        logger.debug("data loaded from %s with shape %s",file_path,df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error("failed to parse the csv file %s",e)
    except FileNotFoundError as e:
        logger.error("file not found %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error occured while loading the data: %s",e)
        raise

def train_model(X_train:np.ndarray,Y_train:np.ndarray,params:dict):
    """
    Train the random forest model
    :param X_train: training features
    :param Y_train: Training labels
    :param params:Dictionary of hyperparameters
    :return trained random forest classifier"""

    try:
        if X_train.shape[0]!=Y_train.shape[0]:
            raise ValueError("the number of sample in x_train and y_train must be the same")
        logger.debug("Initializing random forest model with parameters: %s",params)

        clf=RandomForestClassifier(n_estimators=params["n_estimators"],random_state=params["random_state"])
        logger.debug("Model training started with %d samples",X_train.shape[0])
        clf.fit(X_train,Y_train)

        logger.debug("model training completed")

        return clf
    except ValueError as e:
        logger.error("value error during model training",e)
        raise
    except Exception as e:
        logger.error("Error during model training: %s",e)
        raise

def save_model(model,file_path:str):
    """
    save the train model to a file
    :param model train model object
    :param file_path: path to save the model file"""

    try:
        #ensure the directory exist
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug("Model saved to %s",file_path)
    except FileNotFoundError as e:
        logger.error('file path not found: %s',e)
        raise
    except Exception as e:
        logger.error("error occured while saving the model: %s",e)
        raise

def main():
    try:
        params={"n_estimators":25,"random_state":2}
        train_data=load_data('data/processed/train_tfidf.csv')
        X_train=train_data.iloc[:,:-1].values
        Y_train=train_data.iloc[:,-1].values
        
        clf=train_model(X_train,Y_train,params)

        model_save_path='models/model.pkl'
        save_model(clf,model_save_path)

    except Exception as e:
        logger.error("failed to complete the model building process: %s",e)
        print(f"error{e}")

if __name__=="__main__":
    main()

    