class ModelPreporation:

    # train model
    @staticmethod
    def train_model(classifier, x_train_scaled, y_train):
        return classifier.fit(x_train_scaled, y_train)
