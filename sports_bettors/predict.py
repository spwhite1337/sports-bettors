from sports_bettors.analytics.model.validate import Validate


def predict(league: str = 'nfl', overwrite: bool = False, retrain: bool = False):
    api = Validate(league=league).load_results()
    if api is None:
        api = Validate()
        api.train()
        api.validate()
    elif retrain:
        api.train()
        api.validate()
    api.predict_next_week()


def predict_cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--league', type=str, default='nfl')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--retrain', action='store_true')
    args = parser.parse_args()
    predict(args.league, args.overwrite, args.retrain)
