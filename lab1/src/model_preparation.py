from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

from common import DataStorage as ds
from common import InfoMetric as im


def model_process(data):
    data = shuffle(data, random_state=42)
    
    # separating data into features and target
    x_train = data[['temp_day', 'temp_night', 'precipitation', 'wind']]
    y_train = data['anomaly_day']

    # create model
    model = LogisticRegression(random_state = 42)
    
    # training model
    model.fit(x_train, y_train)

    # prediction on training data
    y_pred = model.predict(x_train)
    
    im.print_metrics(y_train, y_pred)

    return model


def main():
    train_data_preprocessed = ds.read_data('../data/train/train_data_preprocessed.csv')
    model = model_process(train_data_preprocessed)
    ds.save_model('../data/model/', model, 'model')

if __name__ == '__main__':
    main()
