import os

from typing import Tuple
from collections import namedtuple

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pystan

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
                 input_path: str = os.path.join(ROOT_DIR, 'data', 'df_curated.csv'),
                 results_dir: str = os.path.join(ROOT_DIR, 'modeling', 'results'),

                 # Transformation
                 random_effect: str = 'Team',
                 features: str = 'RushOnly',
                 poll: str = 'APTop25Rank',

                 # Modeling
                 response: str = 'Win',
                 iterations: int = 1000,
                 chains: int = 2,
                 verbose: bool = True
                 ):
        # I/O
        self.input_path = input_path
        self.results_dir = results_dir

        # Transformation
        self.random_effect = random_effect.lower()
        self.feature_label = features
        self.features = self.feature_sets[features].features
        self.poll = poll
        self.scales = {}
        self.random_effect_map = {}
        self.random_effect_inv = {}

        # Modeling
        self.response = response
        self.iterations = iterations
        self.chains = chains
        self.verbose = verbose
        self.model = None

        # Quality check on inputs
        assert self.random_effect in self.random_effects
        assert self.poll in self.polls

    @staticmethod
    def etl(input_path: str = None):
        """
        Load data
        """
        logger.info('Loading Curated Data')
        input_path = os.path.join(ROOT_DIR, 'data', 'df_curated.csv') if input_path is None else input_path
        if not os.path.exists(input_path):
            raise FileNotFoundError('No curated data, run `cf_curate`')
        return pd.read_csv(input_path)

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

    def fit_transform(self, df: pd.DataFrame) -> dict:
        """
        Create features and scale
        """
        logger.info('Fitting and Transforming Data')
        # Specify random_effect and map to integer
        df['RandomEffect'] = self._define_random_effect(df)
        groups = sorted(list(set(df['RandomEffect'])))
        self.random_effect_map = dict(zip(groups, range(len(groups))))
        self.random_effect_inv = {v: k for k, v in self.random_effect_map.items()}
        df['RandomEffect'] = df['RandomEffect'].map(self.random_effect_map)

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

        # Convert data to dictionary for Pystan API input
        pystan_data = {feature: df[feature].values for feature in self.features}
        pystan_data['N'] = df.shape[0]
        pystan_data['J'] = len(set(df['RandomEffect']))
        pystan_data['RandomEffect'] = df['RandomEffect'].values + 1
        pystan_data['y'] = df['response'].values

        return pystan_data

    def transform(self, df: pd.DataFrame) -> dict:
        """
        Create features and normalize
        """
        logger.info('Transforming Data.')
        # Specify Random Effect
        df['RandomEffect'] = self._define_random_effect(df)

        # Engineer Features
        df = self._engineer_features(df)

        # Filter if necessary
        df = self.filters[self.response](df)

        # Scale
        if len(self.scales) == 0:
            raise ValueError('Fit model first.')

        for feature in self.features:
            df[feature] = (df[feature] - self.scales[feature][0]) / self.scales[feature][1]

        # Subset and Sort
        df = df[['RandomEffect'] + self.features].sort_values('RandomEffect').reset_index(drop=True)

        # Convert data to dictionary for Pystan API input
        pystan_data = {feature: df[feature].values for feature in self.features}
        pystan_data['N'] = df.shape[0]
        pystan_data['J'] = len(set(df['RandomEffect']))
        pystan_data['RandomEffect'] = df['RandomEffect'].values + 1

        return pystan_data

    def model_code(self):
        """
        Convert list of features into a stan-compatible model code
        """
        variables = ' '.join(['vector[N] {};'.format(feature) for feature in self.features])
        parameters = ' '.join(['real b{};'.format(fdx) for fdx in range(len(self.features))])
        transformation = ' '.join(['+ {}[i] * b{}'.format(feature, fdx) for fdx, feature in enumerate(self.features)])
        model = ' '.join(['b{} ~ normal(0, 1);'.format(fdx) for fdx in range(len(self.features))])
        model_code = """
        data {{
            int<lower=0> J;
            int<lower=0> N;
            int<lower=1, upper=J> RandomEffect[N];
            {variables}
            vector[N] y;
        }}
        parameters {{
            vector[J] a;
            {parameters}
            real mu_a;
            real<lower=0,upper=100> sigma_a;
            real<lower=0,upper=100> sigma_y;
        }}
        transformed parameters {{
            vector[N] y_hat;
            for (i in 1:N)
                y_hat[i] = a[RandomEffect[i]] {transformation};
        }}
        model {{
            sigma_a ~ uniform(0, 100);
            a ~ normal(mu_a, sigma_a);
            {model}
            sigma_y ~ uniform(0, 100);
            y ~ normal(y_hat, sigma_y);
        }}
        """.format(variables=variables, parameters=parameters, transformation=transformation, model=model)

        return model_code

    def fit(self, df: pd.DataFrame = None) -> pystan.stan:
        """
        Fit a pystan model
        """
        logger.info('Fitting a pystan Model')
        if df is None:
            df = self.etl()
        input_data = self.fit_transform(df)
        model_code = self.model_code()

        # Fit stan model
        self.model = pystan.stan(model_code=model_code, data=input_data, iter=self.iterations, chains=self.chains,
                                 verbose=self.verbose, model_name='{}_{}'.format(self.feature_label, self.response),
                                 seed=187)

        return self.model

    def diagnostics(self):
        """
        Print diagnostics for the fit model
        """
        if self.model is None:
            raise ValueError('Fit a model first.')

        # Print r-hats

        # Print distributions of parameters

        # Predict on dataset

        # For Binaries, plot a ROC curve, histogram of predictions by class

        # For continuous, plot a distribution of residuals with r-squared and MSE
