from sports_bettors.analytics.model.validate import Validate


def predict(retrain: bool = False):
    api = Validate().load_results()
    if retrain:
        api.train()
        api.validate()
    api.predict_next_week()


def predict_cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain', action='store_true')
    args = parser.parse_args()
    predict(args.retrain)
