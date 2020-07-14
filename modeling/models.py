import os
from collections import namedtuple

import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from modeling.base import BaseBettingAid
from config import ROOT_DIR, logger

Features = namedtuple('Features', ['label', 'features'])


class CollegeFootballBettingAid(BaseBettingAid):
    """
    Object define parameters for college_football football
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
        'pass_proportion': lambda row: row['passAttempts'] / (row['passAttempts'] + row['rushingAttempts']),
        'total_points': lambda row: row['points'] + row['opp_points']
    }

    # Feature set to use for modeling (each value must be in the curated dataset or as a key in feature_creators)
    feature_sets = {
        'RushOnly': Features('RushOnly', ['rushingYards', 'rushingAttempts']),
        'PassOnly': Features('PassOnly', ['netPassingYards', 'passAttempts']),
        'Offense': Features('Offense', ['rushingYards', 'netPassingYards', 'rushingAttempts', 'passAttempts']),
        'OffenseAdv': Features('OffenseAdv', ['rush_yds_adv', 'pass_yds_adv', 'to_margin']),
        'PlaySelection': Features('PlaySelection', ['pass_proportion', 'fourthDownAttempts']),
        'All': Features('All', ['is_home', 'rush_yds_adv', 'pass_yds_adv', 'penalty_yds_adv', 'ptime_adv', 'to_margin',
                                'firstdowns_adv'])
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

    # I/O
    input_path = os.path.join(ROOT_DIR, 'data', 'college_football', 'df_curated.csv')
    results_dir = os.path.join(ROOT_DIR, 'modeling', 'results', 'college_football')

    # Override this in base because you need to fillna for ranked teams
    def _define_random_effect(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.random_effect == 'ranked_team':
            return df[self.poll].fillna(0).astype(str)
        elif self.random_effect == 'ranked_opponent':
            return df['opp_' + self.poll].fillna(0).astype(str)
        else:
            return df[self.random_effect].astype(str)

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
                plt.hist(residuals, label='MSE: {m:0.3f}'.format(m=mse), bins=20)
                plt.legend()
                plt.xlabel('Residuals')
                plt.ylabel('Counts')
                plt.grid(True)
                plt.tight_layout()
                pdf.savefig()
                plt.close()
