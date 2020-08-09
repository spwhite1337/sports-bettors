import pandas as pd
import numpy as np
from scipy.special import expit


from sports_bettors.api import SportsPredictor


def clean_inputs(inputs: dict) -> dict:
    return inputs


def populate(league: str, feature_set: str, team: str, opponent: str, **kwargs):
    """
    Generate a json friendly dataframe of sports-bettors outputs
    """
    assert league in ['college_football', 'nfl']
    predictor = SportsPredictor(league=league)
    predictor.load()

    records = []
    input_set = {
        'random_effect': 'team',
        'feature_set': feature_set,
        'inputs': {
            'RandomEffect': team
        }
    }
    if feature_set == 'PointsScored':
        for total_points in range(10, 100):
            input_set['inputs']['total_points'] = total_points
            output = predictor.predict(**input_set)[('team', feature_set, 'Win')]
            record = {
                'RandomEffect': team,
                'total_points': total_points,
                'WinLB': expit(output['mu']['lb']),
                'Win': expit(output['mu']['mean']),
                'WinUB': expit(output['mu']['ub'])
            }
            records.append(record)

    elif feature_set == 'RushOnly':
        pass

    return pd.DataFrame.from_records(records)


def points_scored():
    pass
