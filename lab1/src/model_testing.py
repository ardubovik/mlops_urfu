from common import DataStorage as ds
from common import InfoMetric as im


def test_model(data, model):
    
    x_test = data.drop("anomaly_day", axis=1)
    y_test = data["anomaly_day"]

    # prediction on test data
    y_pred = model.predict(x_test)

    # accuracy
    accuracy = model.score(x_test, y_test)

    print(f"accuracy on testing data: {accuracy:.2f}")
    
    # print metrics
    im.print_metrics(y_test, y_pred)


def main():
    data = ds.read_data('../data/test/test_data_preprocessed.csv')
    model = ds.load_model('../data/model/model.pkl')
    
    test_model(data, model)


if __name__ == "__main__":
    main()
