import os
import re
import pickle

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

    # Feature set to use for modeling (each value must be in the curated dataset or as a key in feature_creators)
    feature_sets = {
            'RushOnly': Features('RushOnly', ['rushingYards', 'rushingAttempts']),
            'PassOnly': Features('PassOnly', ['passingYards', 'passingAttempts']),
            'Offense': Features('Offense', ['rushingYards', 'passingYards', 'rushingAttempts', 'passingAttempts']),
            'OffenseAdv': Features('OffenseAdv', ['rush_yds_adv', 'pass_yds_adv', 'to_margin']),
            'PlaySelection': Features('PlaySelection', ['pass_proportion', 'fourthDownAttempts']),
            'All': Features('All', ['is_home', 'rush_yds_adv', 'pass_yds_adv', 'penalty_yds_adv', 'ptime_adv',
                                    'to_margin', 'firstdowns_adv'])
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

    # Model types for each response
    response_distributions = {
        'Win': 'bernoulli_logit',
        'WinMargin': 'linear',
        'LossMargin': 'linear',
        'TotalPoints': 'linear',
        'Margin': 'linear'
    }

    def __init__(self,
                 # I/O
                 input_path: str = os.path.join(ROOT_DIR, 'data', 'df_curated.csv'),
                 results_dir: str = os.path.join(ROOT_DIR, 'modeling', 'results'),
                 version: str = 'v1',

                 # Transformation
                 random_effect: str = 'Team',
                 features: str = 'RushOnly',
                 poll: str = 'APTop25Rank',

                 # Modeling
                 response: str = 'Margin',
                 iterations: int = 1000,
                 chains: int = 2,
                 verbose: bool = True
                 ):
        # I/O
        self.input_path = input_path
        self.results_dir = results_dir
        self.version = version

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

    def etl(self, input_path: str = None):
        """
        Load data
        """
        logger.info('Loading Curated Data')
        input_path = self.input_path if input_path is None else input_path
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
        self.random_effect_map = dict(zip(
            groups,
            ['a[' + str(gdx) + ']' for gdx in range(1, len(groups) + 1)]  # Stan indexes from 1, not 0
        ))
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
        pystan_data.update({
            'N': df.shape[0],
            'J': len(set(df['RandomEffect'])),
            'RandomEffect': df['RandomEffect'].values,
            'y': df['response'].values
        })

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
        pystan_data.update({
            'N': df.shape[0],
            'J': len(set(df['RandomEffect'])),
            'RandomEffect': df['RandomEffect'].values + 1
        })

        return pystan_data

    def model_code(self):
        """
        Convert list of features into a stan-compatible model code
        """
        response = {
            'linear': 'y ~ normal(y_hat, sigma_y)',
            'bernoulli_logit': 'y ~ bernoulli_logit(y_hat)'
        }.get(self.response_distributions[self.response])

        variables = ' '.join(['vector[N] {};'.format(feature) for feature in self.features])
        parameters = ' '.join(['real b{};'.format(fdx) for fdx in range(len(self.features))])
        transformation = ' '.join(['+ {}[i] * b{}'.format(feature, fdx) for fdx, feature in enumerate(self.features)])
        model = ' '.join(['b{} ~ normal(0, 1);'.format(fdx) for fdx in range(len(self.features))])
        model_code = """
        data {{
            int<lower=0> J; int<lower=0> N; int<lower=1, upper=J> RandomEffect[N]; vector[N] y;
            {variables}
        }}
        parameters {{
            vector[J] a; real mu_a; real<lower=0,upper=100> sigma_a; real<lower=0,upper=100> sigma_y;
            {parameters}
        }}
        transformed parameters {{
            vector[N] y_hat;
            for (i in 1:N)
                y_hat[i] = a[RandomEffect[i]] {transformation};
        }}
        model {{
            sigma_a ~ uniform(0, 100); a ~ normal(mu_a, sigma_a); sigma_y ~ uniform(0, 100); {response};
            {model}
        }}
        """.format(variables=variables, parameters=parameters, transformation=transformation, response=response,
                   model=model)

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
                                 verbose=self.verbose, model_name='{}_{}_'.format(self.feature_label, self.response),
                                 seed=187)

        return self.model

    def diagnose(self):
        """
        Print diagnostics for the fit model
        """
        if self.model is None:
            raise ValueError('Fit a model first.')

        # Get model summary
        summary = self.model.summary()
        df_summary = pd.DataFrame(summary['summary'], columns=summary['summary_colnames']).\
            assign(labels=summary['summary_rownames'])

        # Get trues
        y = self.fit_transform(self.etl())['y']
        preds = df_summary[df_summary['labels'].str.contains('y_hat')]['mean'].values

        # Random Intercepts
        df_random_effects = df_summary[df_summary['labels'].str.startswith('a[')]
        df_random_effects['labels'] = df_random_effects['labels'].map(self.random_effect_inv)

        # Coefficients
        df_coefs = df_summary[df_summary['labels'].str.contains('^b[0-9]', regex=True)].assign(labels=self.features)

        # Globals
        df_globals = df_summary[df_summary['labels'].isin(['mu_a', 'sigma_a', 'sigma_y'])]

        # Boxplot distribution of parameters

        # For Binaries, plot a ROC curve, histogram of predictions by class

        # For continuous, plot a distribution of residuals with r-squared and MSE

    def save(self, save_path: str = None):
        """
        Save object
        """
        if save_path is None:
            save_path = 'classifier_{}_{}_{}_{}.pkl'.format(self.feature_label, self.random_effect, self.response,
                                                            self.version)
        if os.path.exists(os.path.join(self.results_dir, save_path)):
            logger.info('WARNING: Overwriting file')
            input('Press enter to continue.')

        with open(os.path.join(self.results_dir, save_path), 'wb') as fp:
            pickle.dump(self, fp)
