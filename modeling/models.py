import os

from typing import Tuple
from collections import namedtuple

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
            'RushOnly': Features('RushOnly', ['rushingYards', 'rushingAttempts']),
            'PassOnly': Features('PassOnly', ['passingYards', 'passingAttempts']),
            'Offense': Features('Offense', ['rushingYards', 'passingYards', 'rushingAttempts', 'passingAttempts']),
            'OffenseAdv': Features('OffenseAdv', ['rush_yds_adv', 'pass_yds_adv', 'to_margin']),
            'PlaySelection': Features('PlaySelection', ['pass_proportion', 'fourthDownAttempts']),
            'All': Features('All', ['is_home', 'rush_yds_adv', 'pass_yds_adv', 'penalty_yds_adv', 'ptime_adv',
                                    'to_margin', 'firstdowns_adv'])
        }

    # Feature Definitions
    feature_creators = {
        'rush_yds_adv': lambda row: row['rushingYards'] - row['opp_rushingYards'],
        'pass_yds_adv': lambda row: row['passingYards'] - row['opp_passingYards'],
        'penalty_yds_adv': lambda row: row['penaltyYards'] - row['opp_penaltyYards'],
        'to_margin': lambda row: row['turnovers'] - row['opp_turnovers'],
        'ptime_adv': lambda row: row['possessionTime'] - row['opp_possessionTime'],
        'firstdowns_adv': lambda row: row['firstDowns'] - row['opp_firstDowns'],
        'pass_proportion': lambda row: row['passAttempts'] / (row['passAttempts'] + row['rushAttempts'])
    }

    # Potential Responses
    responses = ['Win', 'WinMargin', 'LossMargin', 'TotalPoints', 'Margin']

    # Response Definitions
    response_creators = {
        'Win': lambda df_sub: (df_sub['points'] > df_sub['opp_points']).astype(int),
        'WinMargin': lambda df_sub: df_sub['points'] - df_sub['opp_points'],
        'LossMargin': lambda df_sub: df_sub['opp_points'] - df_sub['points'],
        'TotalPoints': lambda df_sub: df_sub['points'] + df_sub['opp_points'],
        'Margin': lambda df_sub: df_sub['points'] - df_sub['opp_points']
    }

    # Filters for select responses
    filters = {
        'Win': lambda df_sub: df_sub,
        'WinMargin': lambda df_sub: df_sub[df_sub['points'] > df_sub['opp_points']],
        'LossMargin': lambda df_sub: df_sub[df_sub['points'] < df_sub['opp_points']],
        'TotalPoints': lambda df_sub: df_sub,
        'Margin': lambda df_sub: df_sub
    }

    def __init__(self,
                 # I/O
                 df_input: pd.DataFrame = None,
                 input_path: str = None,

                 # Transformation
                 random_effect: str = 'Team',
                 features: str = 'RushOnly',
                 poll: str = 'APTop25Rank',

                 # Modeling
                 response: str = 'Win',
                 ):
        # I/O
        self.df_input = df_input if df_input is not None else self.etl(input_path)

        # Transformation
        self.random_effect = random_effect.lower()
        self.features = self.feature_sets[features].features
        self.poll = poll
        self.scales = {}

        # Modeling
        self.response = response

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

    def _define_random_effect(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.random_effect == 'ranked_team':
            return df[self.poll + 'Rank'].fillna(0).astype(str)
        elif self.random_effect == 'ranked_opponent':
            return df['opp_' + self.poll + 'Rank'].fillna(0).astype(str)
        else:
            return df[self.random_effect].astype(str)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature in self.features:
            if feature in self.feature_creators.keys():
                df[feature] = df.apply(lambda row: self.feature_creators[feature](row), axis=1)
        return df

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create features and scale
        """
        # Specify random_effect
        df['RandomEffect'] = self._define_random_effect(df)

        # Engineer features
        df = self._engineer_features(df)

        # Engineer response
        df['response'] = self.response_creators[self.response](df)

        # Filter if necessary
        df = self.filters[self.response](df)

        # Subset and Sort
        df = df[['RandomEffect'] + ['response'] + self.features].sort_values('RandomEffect').reset_index(drop=True)

        # Scale
        for feature in self.features:
            self.scales[feature] = (df[feature].mean(), df[feature].std())
            df[feature] = (df[feature] - self.scales[feature][0]) / self.scales[feature][1]

        return df.drop('response', axis=1), df['response']

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features and normalize
        """
        # Specify Random Effect
        df['RandomEffect'] = self._define_random_effect(df)

        # Engineer Features
        df = self._engineer_features(df)

        # Filter if necessary
        df = self.filters[self.response](df)

        # Scale
        for feature in self.features:
            df[feature] = (df[feature] - self.scales[feature][0]) / self.scales[feature][1]

        # Subset and Sort
        df = df[['RandomEffect'] + self.features].sort_values('RandomEffect').reset_index(drop=True)

        return df
