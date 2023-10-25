from sports_bettors.analytics.model import Model


def predict():
    Model().predict_next_week()


def predict_cli():
    predict()
