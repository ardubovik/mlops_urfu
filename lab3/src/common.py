import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from logger import logger


class DataStorage:

    # save data to csv file
    @staticmethod
    def save_data_to_csv(folder, name, data):
        os.makedirs(folder, exist_ok=True)

        filepath = os.path.join(folder, f'{name}.csv')
        data.to_csv(filepath, index=False)
        logger.debug(f'data saved to {filepath}')

    # read data from file
    @staticmethod
    def read_data(filepath):
        return pd.read_csv(filepath)

    # save model to pkl
    @staticmethod
    def save_model(path, model, name):
        os.makedirs(path, exist_ok=True)

        joblib.dump(model, f'{path}{name}.pkl')
        logger.debug(f'model saved to {path}{name}.pkl')

    # load model from file
    @staticmethod
    def load_model(path):
        return joblib.load(path)
    
    # load dataset
    @staticmethod
    def load_dataset(dataset):
        return dataset()


class InfoMetric:
    
    # print metrics
    @staticmethod
    def print_metrics(y_train, y_pred):

        # calculation metrics
        accuracy = accuracy_score(y_train, y_pred)
        precision = precision_score(y_train, y_pred)
        recall = recall_score(y_train, y_pred)
        f1 = f1_score(y_train, y_pred)

        logger.debug(pd.DataFrame({
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1-score': [f1]
        }))

class DataCreator:

    # split dataset into training and test
    @staticmethod
    def split_data(data, target, test_size, random_state, shuffle = True):
        return train_test_split(
            data,
            target,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )
