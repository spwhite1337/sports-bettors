import os

from collections import namedtuple

import pandas as pd
import numpy as np

from config import ROOT_DIR, logger

Features = namedtuple('Features', ['label', 'features'])


class WinClassifier(object):
    """
    Object to predict binary result of football games
    """
    feature_sets = {
        'Team': [
            Features('RushOnly', ['']),
            Features('PassOnly', ['']),
            Features('TotalOffense', [''])
        ],
        'Opponent': [
            Features('RushOnly', ['']),
            Features('PassOnly', [''])
        ],
        'RankedTeam': [
            Features('RushOnly', ['']),
            Features('PassOnly', [''])
        ]
    }

    def __init__(self, df_input: pd.DataFrame = None, feature_set: str = None):
        self.df_input = df_input

    def etl(self, input_path: str = None):
        """
        Load data
        """
        logger.info('Loading Curated Data')
        input_path = os.path.join(ROOT_DIR, 'data', 'df_curated.csv') if input_path is None else input_path
        if not os.path.exists(input_path):
            raise FileNotFoundError('No curated data, run cf_curate')
        self.df_input = pd.read_csv(input_path)

        return self.df_input

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """

        """



