import os
import re
import pickle
from unittest import TestCase

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sports_bettors.utils.college_football.models import CollegeFootballBettingAid

from config import ROOT_DIR, logger, version


class TestPredictors(TestCase):

    def test_college_predictors(self):
        logger.info('Working on College Football')
        with open(os.path.join(ROOT_DIR, 'modeling', 'results', 'college_football',
                               'predictor_set_{}.pkl'.format(version)), 'rb') as fp:
            predictors = pickle.load(fp)

        # Loop experiments
        for random_effect in CollegeFootballBettingAid.random_effects:
            for feature_set in CollegeFootballBettingAid.feature_sets.keys():
                for response in CollegeFootballBettingAid.responses:
                    # Check if it exists
                    model_path = os.path.join(ROOT_DIR, 'modeling', 'results', 'college_football', response,
                                              feature_set, random_effect, 'aid_{}.pkl'.format(version))
                    if not os.path.exists(model_path):
                        logger.info('WARNING: No model for {}, {}, {}'.format(random_effect, feature_set, response))
                        continue

                    # Load betting aid
                    with open(model_path, 'rb') as fp:
                        aid = pickle.load(fp)

                    logger.info('Load Data')
                    df_data = aid.etl()

                    # Transform the data but don't scale it
                    logger.info('Transform Data')
                    data = aid.fit_transform(df_data, skip_scaling=True)
                    # Drop params for pystan
                    data.pop('N')
                    data.pop('J')
                    df = pd.DataFrame.from_dict(data)

                    logger.info('Get predictions from pystan summary')
                    df['y_fit'] = aid.summary[aid.summary['labels'].str.contains('y_hat')]['mean'].values
                    df['y_fit_lb'] = df['y_fit'] - aid.summary[aid.summary['labels'].str.contains('y_hat')]['sd'].values
                    df['y_fit_ub'] = df['y_fit'] + aid.summary[aid.summary['labels'].str.contains('y_hat')]['sd'].values

                    # Generate preds
                    logger.info('Generate Predictions')
                    predictor = predictors[(random_effect, feature_set, response)]
                    df['y_preds'] = df[['RandomEffect'] + aid.features].apply(lambda r: predictor(r)['mean'], axis=1)
                    df['y_preds_lb'] = df[['RandomEffect'] + aid.features].apply(lambda r: predictor(r)['lb'], axis=1)
                    df['y_preds_ub'] = df[['RandomEffect'] + aid.features].apply(lambda r: predictor(r)['ub'], axis=1)

                    # Save
                    save_dir = os.path.join(ROOT_DIR, 'tests', 'results', response, feature_set, random_effect)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    # Diagnostics
                    with PdfPages(os.path.join(save_dir, 'college_football_test.pdf')) as pdf:
                        # Scatter plot from each source
                        df_sample = df.sample(min(df.shape[0], 1000))
                        lb = min([df_sample['y_fit'].min(), df_sample['y_preds'].min()])
                        ub = max([df_sample['y_fit'].max(), df_sample['y_preds'].max()])
                        plt.figure(figsize=(8, 8))
                        plt.scatter(df_sample['y_fit'], df_sample['y_preds'], alpha=0.5)
                        plt.plot([lb, ub], [lb, ub], color='black', linestyle='dashed')
                        plt.xlabel('Pystan Predictions')
                        plt.ylabel('Predictor')
                        plt.title('Predictions Test.')
                        plt.grid(True)
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()

                        # Scatter plot of errors
                        df_sample = df.sample(min(df.shape[0], 1000))
                        lb = min([df_sample['y_fit_lb'].min(), df_sample['y_preds_lb'].min()])
                        ub = max([df_sample['y_fit_lb'].max(), df_sample['y_preds_lb'].max()])
                        plt.figure(figsize=(8, 8))
                        plt.scatter(df_sample['y_fit_lb'], df_sample['y_preds_lb'], alpha=0.5)
                        plt.plot([lb, ub], [lb, ub], color='black', linestyle='dashed')
                        plt.xlabel('Pystan LB (approx)')
                        plt.ylabel('Predictor LB (approx)')
                        plt.title('Lower Bounds')
                        plt.grid(True)
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()

                        df_sample = df.sample(min(df.shape[0], 1000))
                        lb = min([df_sample['y_fit_ub'].min(), df_sample['y_preds_ub'].min()])
                        ub = max([df_sample['y_fit_ub'].max(), df_sample['y_preds_ub'].max()])
                        plt.figure(figsize=(8, 8))
                        plt.scatter(df_sample['y_fit_ub'], df_sample['y_preds_ub'], alpha=0.5)
                        plt.plot([lb, ub], [lb, ub], color='black', linestyle='dashed')
                        plt.xlabel('Pystan UB (approx)')
                        plt.ylabel('Predictor UB (approx)')
                        plt.title('Upper Bounds')
                        plt.grid(True)
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()

                        # Histogram of residuals
                        plt.figure(figsize=(8, 8))
                        plt.hist(df['y_fit'] - df['y_preds'], alpha=0.5, bins=20)
                        plt.xlabel('Residual (Pystan - predictor)')
                        plt.ylabel('Count')
                        plt.title('Residuals of Predictions')
                        plt.grid(True)
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()

    def test_custom_college(self):
        with open(os.path.join(ROOT_DIR, 'modeling', 'results', 'college_football',
                               'predictor_set_{}.pkl'.format(version)), 'rb') as fp:
            predictors = pickle.load(fp)

        # Good rushing game for Iowa
        iowa = {
            'RandomEffect': 'Iowa',
            'rushingYards': 150,
            'rushingAttempts': 30
        }
        for response in CollegeFootballBettingAid.responses:
            if ('team', 'RushOnly', response) not in predictors.keys():
                continue
            output = predictors[('team', 'RushOnly', response)](iowa)
            logger.info('{}: {}'.format(response, output))
