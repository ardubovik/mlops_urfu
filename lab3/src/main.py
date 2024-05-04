import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from common import DataStorage as ds
from common import DataCreator as dc
from model_preprocessor import ModelPreprecessor as mp
from model_preparation import ModelPreporation as mpp
from logger import logger

def main():
    # load dataset
    iris = ds.load_dataset(load_iris)
    logger.debug('dataset loaded')

    # save dataset
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    logger.debug('saving dataset')
    ds.save_data_to_csv('../data/datasets', 'dataset', iris_df)

    # split dataset into training and test
    x_train, y, y_train, yy = dc.split_data(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )
    logger.debug('dataset splitted')

    # preprocess data
    scaler = StandardScaler()
    x_train_scaled = mp.fit_transform(x_train, scaler)
    logger.debug('data preprocessed')

    # train model
    model = mpp.train_model(
        RandomForestClassifier(n_estimators=100, random_state=42),
        x_train_scaled,
        y_train,
    )
    logger.debug('model trained')

    # save model
    ds.save_model('../data/models/', model, 'model')

    # save scaler
    ds.save_model('../data/models/', scaler, 'scaler')

if __name__ == '__main__':
    main()
