from typing import Optional
from sports_bettors.analytics.model.validate import Validate


def predict(league: Optional[str] = None, overwrite: bool = False):
    leagues = ['nfl', 'college_football'] if league is None else [league]
    for league in leagues:
        api = Validate(league=league).load_results()
        if api is None or overwrite:
            api = Validate()
            api.train()
            api.validate()
        api.predict_next_week()


def predict_cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--league', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    predict(args.league, args.overwrite)
