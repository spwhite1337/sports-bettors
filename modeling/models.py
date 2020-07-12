import os
import re
import pickle

from collections import namedtuple

import pandas as pd
import numpy as np

import pystan
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from config import ROOT_DIR, logger, version

Features = namedtuple('Features', ['label', 'features'])


class FootballPredictor(object):
    """
    Lightweight predictor that accepts a dictionary of features (name: val) and random_effect ('RandomEffect': val)
    And returns
    """

    def __init__(self, scales: dict, predictor: dict):
        self.scales = scales
        self.predictor = predictor

    def predict(self, data: dict) -> dict:
        """
        Scale and predict from input data
        """
        # Get random effect
        random_effect = data.pop('RandomEffect')

        # Scale data
        data = {feature: (val - self.scales[feature][0]) / self.scales[feature][1] for feature, val in data.items()}

        # Get output including mean, ub, and lb
        output = {
            'lb': self.predictor['intercept'][0] +
                  self.predictor['random_effect'].get(random_effect, (0, 0, 0))[0] +
                  np.sum([
                      np.min([c * v for c in self.predictor['coefficients'][f]]) for f, v in data.items()
                  ]) + self.predictor.get('noise', (0, 0, 0))[0],
            'mean': self.predictor['intercept'][1] +
                    self.predictor['random_effect'].get(random_effect, (0, 0, 0))[1] +
                    np.sum([self.predictor['coefficients'][f][1] * v for f, v in data.items()]) +
                    self.predictor.get('noise', (0, 0, 0))[1],
            'ub': self.predictor['intercept'][2] +
                  self.predictor['random_effect'].get(random_effect, (0, 0, 0))[1] +
                  np.sum([
                      np.max([c * v for c in self.predictor['coefficients'][f]]) for f, v in data.items()
                  ]) + self.predictor.get('noise', (0, 0, 0))[2],
        }

        return output


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
        'pass_yds_adv': lambda row: row['netPassingYards'] - row['opp_netPassingYards'],
        'penalty_yds_adv': lambda row: row['penaltyYards'] - row['opp_penaltyYards'],
        'to_margin': lambda row: row['turnovers'] - row['opp_turnovers'],
        'ptime_adv': lambda row: row['possessionTime'] - row['opp_possessionTime'],
        'firstdowns_adv': lambda row: row['firstDowns'] - row['opp_firstDowns'],
        'pass_proportion': lambda row: row['passAttempts'] / (row['passAttempts'] + row['rushingAttempts'])
    }

    # Feature set to use for modeling (each value must be in the curated dataset or as a key in feature_creators)
    feature_sets = {
        'RushOnly': Features('RushOnly', ['rushingYards', 'rushingAttempts']),
        'PassOnly': Features('PassOnly', ['netPassingYards', 'passAttempts']),
        'Offense': Features('Offense', ['rushingYards', 'netPassingYards', 'rushingAttempts', 'passAttempts']),
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
        self.input_path = input_path
        self.results_dir = os.path.join(results_dir, response, features, random_effect)
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
        assert self.poll in self.polls

        # Make dirs
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
                )),
                'intercept': (intercept - intercept_sd, intercept, intercept + intercept_sd),
        }

        # Add noise is applicable
        if self.response_distributions[self.response] == 'bernoulli_logit':
            predictor['noise'] = (noise - noise_sd, noise, noise + noise_sd)

        # Define predictor object
        self.predictor = FootballPredictor(scales=self.scales, predictor=predictor)

        return self.model, self.summary

    def diagnose(self):
        """
        Print diagnostics for the fit model
        """
        if self.summary is None:
            raise ValueError('Fit a model first.')

        logger.info('Printing Results.')
        # Get trues
        y = self.fit_transform(self.etl())['y']
        preds = self.summary[self.summary['labels'].str.contains('y_hat')]['mean'].values

        # Random Intercepts
        df_random_effects = self.summary[self.summary['labels'].str.startswith('a[')]. \
            sort_values('mean', ascending=False).reset_index(drop=True)
        df_random_effects['labels'] = df_random_effects['labels'].map(self.random_effect_inv)

        # Coefficients
        df_coefs = self.summary[self.summary['labels'].str.contains('^b[0-9]', regex=True)]. \
            assign(labels=self.features). \
            sort_values('mean', ascending=False). \
            reset_index(drop=True)

        # Globals
        df_globals = self.summary[self.summary['labels'].isin(['mu_a', 'sigma_a', 'sigma_y'])].reset_index(drop=True)
        if self.response_distributions[self.response] == 'bernoulli_logit':
            df_globals = df_globals[df_globals['labels'] != 'sigma_y'].reset_index(drop=True)

        with PdfPages(os.path.join(self.results_dir, 'diagnostics_{}.pdf'.format(self.version))) as pdf:
            # Bar graph of random effects for top 10, bottom 10, big10 teams
            plt.figure(figsize=(8, 8))
            df_top10 = df_random_effects.sort_values('mean', ascending=False).head(10).reset_index(drop=True)
            plt.bar(df_top10['labels'], df_top10['mean'])
            plt.errorbar(x=df_top10.index, y=df_top10['mean'], yerr=df_top10['sd'], fmt='none', ecolor='black')
            plt.grid(True)
            plt.xticks(rotation=90)
            plt.title('Top Ten Teams')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Bottom 10
            plt.figure(figsize=(8, 8))
            df_bot10 = df_random_effects.sort_values('mean', ascending=False).tail(10).reset_index(drop=True)
            plt.bar(df_bot10['labels'], df_bot10['mean'])
            plt.errorbar(x=df_bot10.index, y=df_bot10['mean'], yerr=df_bot10['sd'], fmt='none', ecolor='black')
            plt.grid(True)
            plt.title('Bottom Ten Teams')
            plt.xticks(rotation=90)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            if self.random_effect in ['team', 'opponent']:
                # Big10
                plt.figure(figsize=(8, 8))
                df_big10 = df_random_effects[df_random_effects['labels'].isin([
                    'Iowa', 'Wisconsin', 'Michigan', 'MichiganState', 'OhioState', 'Indiana', 'Illinois', 'Nebraska',
                    'PennState', 'Minnesota', 'Rutgers', 'Maryland'
                ])].sort_values('mean', ascending=False).reset_index(drop=True)
                plt.bar(df_big10['labels'], df_big10['mean'])
                plt.errorbar(x=df_big10.index, y=df_big10['mean'], yerr=df_big10['sd'], fmt='none', ecolor='black')
                plt.grid(True)
                plt.hlines(xmax=max(df_big10.index) + 1, xmin=min(df_big10.index) - 1, y=0, linestyles='dashed')
                plt.title('Big Ten Teams')
                plt.xticks(rotation=90)
                plt.tight_layout()
                pdf.savefig()
                plt.close()

            # Coefficients
            plt.figure(figsize=(8, 8))
            plt.bar(df_coefs['labels'], df_coefs['mean'])
            plt.errorbar(x=df_coefs.index, y=df_coefs['mean'], yerr=df_coefs['sd'], fmt='none', ecolor='black')
            plt.grid(True)
            plt.hlines(xmax=max(df_coefs.index) + 1, xmin=min(df_coefs.index) - 1, y=0, linestyles='dashed')
            plt.title('Coefficients')
            plt.xticks(rotation=90)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            # Globals
            plt.figure(figsize=(8, 8))
            plt.bar(df_globals['labels'], df_globals['mean'])
            plt.errorbar(x=df_globals.index, y=df_globals['mean'], yerr=df_globals['sd'], fmt='none', ecolor='black')
            plt.grid(True)
            plt.hlines(xmax=max(df_globals.index) + 1, xmin=min(df_globals.index) - 1, y=0, linestyles='dashed')
            plt.title('Globals')
            plt.xticks(rotation=90)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            if self.response_distributions[self.response] == 'bernoulli_logit':
                logger.info('Extra diagnostics for {}'.format(self.response))
                # For Binaries, plot a ROC curve, histogram of predictions by class
                fpr, tpr, th = roc_curve(y, preds)
                score = auc(fpr, tpr)
                plt.figure(figsize=(8, 8))
                plt.plot(fpr, tpr, label='AUC: {a:0.3f}'.format(a=score))
                plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')
                plt.grid(True)
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend()
                plt.tight_layout()
                pdf.savefig()
                plt.close()

                # Precision / Recall
                plt.figure(figsize=(8, 8))
                plt.plot(th, fpr, label='False Positive Rate')
                plt.plot(th, tpr, label='True Positive Rate')
                plt.title('Precisions / Recall')
                plt.xlabel('Cutoff')
                plt.ylabel('Rate')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                pdf.savefig()
                plt.close()

                # Histograms
                bins = np.linspace(min(preds) * 0.99, max(preds) * 1.01, 20)
                plt.figure(figsize=(8, 8))
                plt.hist(preds[y == 1], alpha=0.5, bins=bins, color='darkorange', density=True, label='Positive')
                plt.hist(preds[y == 0], alpha=0.5, bins=bins, density=True, label='Negative')
                plt.grid(True)
                plt.legend()
                plt.xlabel('Probability')
                plt.ylabel('Density')
                plt.tight_layout()
                pdf.savefig()
                plt.close()

            elif self.response_distributions[self.response] == 'linear':
                logger.info('Extra Diagnostics for {}'.format(self.response))
                # For continuous, plot a distribution of residuals with r-squared and MSE
                residuals = y - preds
                mse = np.sum(residuals ** 2)
                plt.figure(figsize=(8, 8))
                plt.hist(residuals, label='MSE: {m:0.3f}'.format(m=mse))
                plt.legend()
                plt.xlabel('Residuals')
                plt.ylabel('Counts')
                plt.grid(True)
                plt.tight_layout()
                pdf.savefig()
                plt.close()

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
