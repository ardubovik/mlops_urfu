class ModelPreprecessor:

    # fit_transform
    @staticmethod
    def fit_transform(data, scaler):
        return scaler.fit_transform(data)

    # transform
    @staticmethod
    def transform(data, scaler):
        return scaler.transform(data)
