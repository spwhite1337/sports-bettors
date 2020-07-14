import os
import re
import pickle

from typing import Tuple
from collections import namedtuple

import pandas as pd
import numpy as np

import pystan


from config import ROOT_DIR, logger, version


Features = namedtuple('Features', ['label', 'features'])


class BetPredictor(object):
    """
    Lightweight predictor that accepts a dictionary of features (name: val) and random_effect ('RandomEffect': val)
    And returns
    """

    def __init__(self, scales: dict, predictor: dict, re_params: Tuple[float, float]):
        self.scales = scales
        self.predictor = predictor
        self.re_params = re_params

    def predict(self, data: dict) -> dict:
        """
        Scale and predict from input data
        """
        # Get random effect
        random_effect = data.pop('RandomEffect')
        re_vals = self.predictor['random_effect'].get(random_effect, (
            self.re_params[0] - self.re_params[1], self.re_params[0], self.re_params[0] + self.re_params[1]
        ))

        # Scale data
        data = {feature: (val - self.scales[feature][0]) / self.scales[feature][1] for feature, val in data.items()}

        # Get output including mean, ub, and lb
        output = {
            'lb': re_vals[0] + np.sum([
                np.min([c * v for c in self.predictor['coefficients'][f]]) for f, v in data.items()
            ]) + self.predictor.get('noise', (0, 0, 0))[0],
            'mean': re_vals[1] + np.sum([
                self.predictor['coefficients'][f][1] * v for f, v in data.items()
            ]) + self.predictor.get('noise', (0, 0, 0))[1],
            'ub': re_vals[2] + np.sum([
                np.max([c * v for c in self.predictor['coefficients'][f]]) for f, v in data.items()
            ]) + self.predictor.get('noise', (0, 0, 0))[2],
        }

        return output


class BaseBettingAid(object):
    """
    Base object for football betting
    """
    # Random effect in hierarchical model.
    random_effects = ['g1', 'g2', 'g3']

    # Feature Definitions
    feature_creators = {
        'x3': lambda row: row['x1'] - row['x2'],
    }

    # Feature set to use for modeling (each value must be in the curated dataset or as a key in feature_creators)
    feature_sets = {
        'univariate': Features('univariate', ['x1']),
        'multivariate': Features('multivariate', ['x1', 'x2']),
        'All': Features('All', ['x1', 'x2', 'x3'])
    }

    # Potential Responses
    responses = ['y1', 'y2']

    # Response Definitions
    response_creators = {
        'y1': lambda df_sub: (df_sub['y1'] > 0.5).astype(int),
        'y2': lambda df_sub: df_sub['y2'],
    }

    # Filters for select responses
    filters = {
        'y1': lambda df_sub: df_sub,
        'y2': lambda df_sub: df_sub,
    }

    # Model types for each response
    response_distributions = {
        'y1': 'bernoulli_logit',
        'y2': 'linear',
    }

    # I/O
    input_path = os.path.join(ROOT_DIR, 'data', 'df_curated.csv')
    results_dir = os.path.join(ROOT_DIR, 'modeling', 'results')

    def __init__(self,
                 # I/O
                 version: str = version,

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
        self.summary = None
        self.predictor = None

        # Quality check on inputs
        assert self.random_effect in self.random_effects

        # Make dirs
        self.results_dir = os.path.join(self.results_dir, response, features, random_effect)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        logger.info('Initialized with {}, {} for {}'.format(self.feature_label, self.random_effect, self.response))

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
        # Engineer features
        df = self._engineer_features(df)

        # Engineer response
        df['response'] = self.response_creators[self.response](df)

        # Filter if necessary
        df = self.filters[self.response](df).copy()

        # Specify random_effect and map to integer
        df['RandomEffect'] = self._define_random_effect(df)
        groups = sorted(list(set(df['RandomEffect'])))
        self.random_effect_map = dict(zip(groups, range(1, len(groups) + 1)))  # Stan indexes from 1, not 0
        self.random_effect_inv = {'a[' + str(v) + ']': k for k, v in self.random_effect_map.items()}
        df['RandomEffect'] = df['RandomEffect'].map(self.random_effect_map)

        # Subset and Sort
        df = df[['RandomEffect'] + ['response'] + self.features].sort_values('RandomEffect').reset_index(drop=True)

        # Drop nas
        df = df.dropna(axis=0).reset_index(drop=True)

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
        # Filter if necessary
        df = self.filters[self.response](df)

        # Specify Random Effect
        df['RandomEffect'] = self._define_random_effect(df)

        # Engineer Features
        df = self._engineer_features(df)

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
        response_var = {
            'linear': 'vector[N] y',
            'bernoulli_logit': 'int<lower=0,upper=1> y[N]'
        }.get(self.response_distributions[self.response])
        variables = ' '.join(['vector[N] {};'.format(feature) for feature in self.features])
        parameters = ' '.join(['real b{};'.format(fdx) for fdx in range(len(self.features))])
        transformation = ' '.join(['+ {}[i] * b{}'.format(feature, fdx) for fdx, feature in enumerate(self.features)])
        model = ' '.join(['b{} ~ normal(0, 1);'.format(fdx) for fdx in range(len(self.features))])
        model_code = """
        data {{
            int<lower=0> J; int<lower=0> N; int<lower=1, upper=J> RandomEffect[N]; {response_var};
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
        """.format(response_var=response_var, variables=variables, parameters=parameters, transformation=transformation,
                   response=response, model=model)

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
                                 verbose=self.verbose,
                                 model_name='{}_{}_{}'.format(self.feature_label, self.random_effect, self.response),
                                 seed=187)
        # Get model summary
        logger.info('Getting model summary for diagnostics')
        summary = self.model.summary()
        self.summary = pd.DataFrame(summary['summary'], columns=summary['summary_colnames']). \
            assign(labels=summary['summary_rownames'])

        # Random Effects
        df_re = self.summary[self.summary['labels'].str.startswith('a[')].reset_index(drop=True)
        df_re['labels'] = df_re['labels'].map(self.random_effect_inv)

        # Coefficients
        df_coefs = self.summary[self.summary['labels'].str.contains('^b[0-9]', regex=True)]. \
            assign(labels=self.features)

        # Global intercept
        intercept = self.summary[self.summary['labels'] == 'mu_a']['mean'].iloc[0]
        intercept_sd = self.summary[self.summary['labels'] == 'mu_a']['sd'].iloc[0]

        # Noise
        noise = self.summary[self.summary['labels'] == 'sigma_y']['mean'].iloc[0]
        noise_sd = self.summary[self.summary['labels'] == 'sigma_y']['sd'].iloc[0]

        # Convert to bare-model for API predictions
        predictor = {
            'random_effect': dict(zip(
                df_re['labels'],
                list(zip(df_re['mean'] - df_re['sd'], df_re['mean'], df_re['mean'] + df_re['sd']))
            )),
            'coefficients': dict(zip(
                df_coefs['labels'],
                list(zip(df_coefs['mean'] - df_coefs['sd'], df_coefs['mean'], df_coefs['mean'] + df_coefs['sd']))
            ))
        }

        # Add noise is applicable
        if self.response_distributions[self.response] != 'bernoulli_logit':
            predictor['noise'] = (noise - noise_sd, noise, noise + noise_sd)

        # Define predictor object
        self.predictor = BetPredictor(scales=self.scales, predictor=predictor, re_params=(intercept, intercept_sd))

        return self.model, self.summary

    def save(self, save_path: str = None):
        """
        Save object
        """
        if save_path is None:
            save_path = 'classifier_{}.pkl'.format(self.version)
        if os.path.exists(os.path.join(self.results_dir, save_path)):
            logger.info('WARNING: Overwriting file')
            input('Press enter to continue.')

        logger.info('Saving object to {}'.format(save_path))
        with open(os.path.join(self.results_dir, save_path), 'wb') as fp:
            pickle.dump(self, fp)

        predictor_path = re.sub('classifier', 'predictor', save_path)
        logger.info('Saving predictor to {}'.format(predictor_path))
        with open(os.path.join(self.results_dir, predictor_path), 'wb') as fp:
            pickle.dump(self.predictor, fp)
