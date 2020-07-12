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
                    with open(os.path.join(ROOT_DIR, 'modeling', 'results', 'classifier_{}_{}_{}_{}.pkl'.format(
                            feature_set, random_effect, response, version)), 'rb') as fp:
                        aid = pickle.load(fp)


                    logger.info('Select Predictor')
                    predictor = predictors[(random_effect, feature_set, response)]

