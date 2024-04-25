import os
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class DataStorage:

    # save data to csv file
    @staticmethod
    def save_data_to_csv(folder, name, data):
        os.makedirs(folder, exist_ok=True)

        filepath = os.path.join(folder, f'{name}.csv')
        data.to_csv(filepath, index=False)
        print(f'data saved to {filepath}')

    # read data from file
    @staticmethod
    def read_data(filepath):
        return pd.read_csv(filepath)

    # save model to pkl
    @staticmethod
    def save_model(path, model, name):
        os.makedirs(path, exist_ok=True)

        joblib.dump(model, f'{path}{name}.pkl')
        print('model trained and saved')

    # load model from file
    @staticmethod
    def load_model(path):
        return joblib.load(path)


class InfoMetric:
    
    # print metrics
    @staticmethod
    def print_metrics(y_train, y_pred):

        # calculation metrics
        accuracy = accuracy_score(y_train, y_pred)
        precision = precision_score(y_train, y_pred)
        recall = recall_score(y_train, y_pred)
        f1 = f1_score(y_train, y_pred)

        print(pd.DataFrame({
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1-score': [f1]
        }))
