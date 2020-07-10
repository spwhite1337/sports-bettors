import os

from typing import Tuple
from collections import namedtuple

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from config import ROOT_DIR, logger

Features = namedtuple('Features', ['label', 'features'])


class FootballBettingAid(object):
    """
    Object to predict binary result of football games
    """
    # Random effect in hierarchical model. One can specify either the team or the opponent name; or they can specify
    # The teams rank or the opponent's rank.
    random_effects = ['team', 'opponent', 'ranked_team', 'ranked_opponent']

    # Poll to use when determining rank
    polls = ['APTop25Rank', 'BCSStandingsRank', 'CoachesPollRank']

    # Feature set to use for modeling
    feature_sets = {
            'RushOnly': Features('RushOnly', ['rush_yds', 'rush_atmps']),
            'PassOnly': Features('PassOnly', ['pass_yds', 'pass_atmps']),
            'Offense': Features('Offense', ['rush_yds', 'pass_yds', 'rush_atmps', 'pass_atmps']),
            'OffenseAdv': Features('OffenseAdv', ['rush_yds_adv', 'pass_yds_adv', 'to_margin']),
            'PlaySelection': Features('PlaySelection', ['pass_proportion', 'fourth_down_attempts']),
            'All': Features('All', ['is_home', 'rush_yds_adv', 'pass_yds_adv', 'penalty_yds_adv', 'ptime_adv',
                                    'to_margin', 'firstdowns_adv'])
        }

    # Potential Responses
    responses = ['Win', 'WinMargin', 'LossMargin', 'TotalPoints', 'Margin']

    def __init__(self,
                 # I/O
                 df_input: pd.DataFrame = None,
                 input_path: str = None,

                 # Transformation
                 random_effect: str = 'Team',
                 features: str = 'RushOnly',
                 poll: str = 'APTop25Rank',

                 # Modeling

                 ):
        # I/O
        self.df_input = df_input if df_input is not None else self.etl(input_path)

        # Transformation
        self.random_effect = random_effect.lower()
        self.features = self.feature_sets[features].features
        self.poll = poll

        # Modeling

        # Quality check on inputs
        assert self.random_effect in self.random_effects
        assert self.poll in self.polls

    def etl(self, input_path: str = None):
        """
        Load data and
        """
        logger.info('Loading Curated Data')
        input_path = os.path.join(ROOT_DIR, 'data', 'df_curated.csv') if input_path is None else input_path
        if not os.path.exists(input_path):
            raise FileNotFoundError('No curated data, run `cf_curate`')
        self.df_input = pd.read_csv(input_path)

        return self.df_input

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create features and scale
        """
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features and normalize
        """
        pass
