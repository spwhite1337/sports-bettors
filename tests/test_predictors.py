import os
import pickle
from unittest import TestCase

from modeling.models import FootballBettingAid

from config import ROOT_DIR, logger, version


class TestPredictors(TestCase):

    def test_predictors(self):
        with open(os.path.join(ROOT_DIR, 'modeling', 'results', 'predictor_set.pkl'), 'rb') as fp:
            predictors = pickle.load(fp)

        # Loop experiments
        for random_effect in FootballBettingAid.random_effects:
            for feature_set in FootballBettingAid.feature_sets.keys():
                for response in FootballBettingAid.responses:
                    logger.info('Load preds from betting aid: {}, {}, {}.'.format(feature_set, random_effect, response))
                    with open(os.path.join(ROOT_DIR, 'modeling', 'results', response, feature_set, random_effect,
                                           'classifier_{}.pkl'.format(version)), 'rb') as fp:
                        aid = pickle.load(fp)

                    logger.info('Load Data')
                    df_data = aid.etl()
                    logger.info('Filter Data')
                    df = aid.filters[aid.response](df_data).copy()
                    logger.info('Get True values')
                    df['y_true'] = aid.fit_transform(df_data)['y']
                    logger.info('Get predictions from pystan')
                    df['y_aid'] = aid.summary[aid.summary['labels'].str.contains('y_hat')]['mean'].values

                    logger.info('Select Predictor')
                    predictor = predictors[(random_effect, feature_set, response)]
                    # Wrangle curated into dict for predictor input
                    df['y_preds'] = None

    def test_custom(self):
        # TODO make a custom input and check output as an example of the API
        pass

