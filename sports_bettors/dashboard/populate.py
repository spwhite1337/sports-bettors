import os
import pickle

import pandas as pd
import numpy as np


from sports_bettors.api import SportsPredictor


def clean_inputs(inputs: dict) -> dict:
    return inputs


def populate(league: str, feature_set: str, random_effect: str, random_effect_vals: list, output_type: str):
    """
    Generate a json friendly dataframe of sports-bettors outputs
    """
    assert league in ['college_football', 'nfl']

    predictor = SportsPredictor(league=league)
    predictor.load()

    records = []
    for random_effect_val in random_effect_vals:
        input_set = {
            'random_effect': random_effect,
            'feature_set': feature_set,
            'inputs': {
                'RandomEffect': random_effect_val
            }
        }
        if feature_set == 'PointsScored':
            for total_points in range(10, 100):
                input_set['inputs']['total_points'] = total_points

                output = predictor.predict(**input_set)[(random_effect, feature_set, 'Win')]
                record = {
                    'RandomEffect': random_effect_val,
                    'total_points': total_points,
                    'WinLB': output['mu']['lb'],
                    'Win': output['mu']['mean'],
                    'WinUB': output['mu']['ub']
                }

                records.append(record)

    print(len(records))

    return pd.DataFrame.from_records(records).to_json()

