from sklearn.preprocessing import StandardScaler

from common import DataStorage as ds


# preprocess data
def preprocess_data(data):
    return StandardScaler().fit_transform(data)


def main():
    train_data = ds.read_data('../data/train/train_data.csv')
    test_data = ds.read_data('../data/test/test_data.csv')

    train_data[[
        'temp_day',
        'temp_night',
        'precipitation',
        'wind',
    ]] = preprocess_data(train_data[[
        'temp_day',
        'temp_night',
        'precipitation',
        'wind',
    ]])

    test_data[[
        'temp_day',
        'temp_night',
        'precipitation',
        'wind',
    ]] = preprocess_data(test_data[[
        'temp_day',
        'temp_night',
        'precipitation',
        'wind',
    ]])

    ds.save_data_to_csv('../data/train', 'train_data_preprocessed', train_data)
    ds.save_data_to_csv('../data/test', 'test_data_preprocessed', test_data)


if __name__ == '__main__':
    main()
