import os
import pickle
from unittest import TestCase

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from modeling.models import CollegeFootballBettingAid

from config import ROOT_DIR, logger, version


class TestPredictors(TestCase):

    def test_college_predictors(self):
        with open(os.path.join(ROOT_DIR, 'modeling', 'results', 'college_football', 'predictor_set.pkl'), 'rb') as fp:
            predictors = pickle.load(fp)

        # Loop experiments
        for random_effect in CollegeFootballBettingAid.random_effects:
            for feature_set in CollegeFootballBettingAid.feature_sets.keys():
                for response in CollegeFootballBettingAid.responses:
                    logger.info('Load preds from betting aid: {}, {}, {}.'.format(feature_set, random_effect, response))
                    with open(os.path.join(ROOT_DIR, 'modeling', 'results', 'college_football', response, feature_set, 
                                           random_effect, 'classifier_{}.pkl'.format(version)), 'rb') as fp:
                        aid = pickle.load(fp)

                    logger.info('Load Data')
                    df_data = aid.etl()

                    logger.info('Transform Data')
                    df_data['RandomEffect'] = aid._define_random_effect(df_data)  # These should be "raw" inputs
                    df = aid.filters[aid.response](df_data).copy()
                    df = aid._engineer_features(df)
                    df['y_true'] = aid.response_creators[aid.response](df)

                    logger.info('Get predictions from pystan')
                    df['y_aid'] = aid.summary[aid.summary['labels'].str.contains('y_hat')]['mean'].values

                    # Generate preds
                    logger.info('Generate Prediction')
                    predictor = predictors[(random_effect, feature_set, response)]
                    df['y_preds'] = df.apply(lambda row: predictor(row)['mean'], axis=1)

                    with PdfPages(os.path.join(ROOT_DIR, 'tests', 'college_football_test.pdf')) as pdf:
                        # Scatter plot from each source
                        plt.figure(figsize=(8, 8))
                        plt.plot(df['y_aid'], df['y_preds'], alpha=0.5)
                        plt.xlabel('Pystan Predictions')
                        plt.ylabel('Predictor')
                        plt.title('Predictions Test.')
                        plt.grid(True)
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()

                        # Histogram of residuals
                        plt.figure(figsize=(8, 8))
                        plt.hist(df['y_aid'] - df['y_preds'], alpha=0.5, bins=20)
                        plt.xlabel('Residual (Pystan - predictor)')
                        plt.ylabel('Count')
                        plt.title('Residuals of Predictions')
                        plt.grid(True)
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()

    def test_custom_college(self):
        with open(os.path.join(ROOT_DIR, 'modeling', 'results', 'college_football', 'predictor_set.pkl'), 'rb') as fp:
            predictors = pickle.load(fp)

        # Good rushing game for Iowa
        iowa = {
            'RandomEffect': 'Iowa',
            'rushingYards': 150,
            'rushingAttempts': 30
        }

        for response in CollegeFootballBettingAid.responses:
            output = predictors[('team', 'RushOnly', response)](iowa)
            logger.info('{}: {}'.format(response, output))
