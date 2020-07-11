import os
import pickle
from unittest import TestCase

from config import ROOT_DIR, logger


class TestPredictors(TestCase):

    def test_predictors(self):
        with open(os.path.join(ROOT_DIR, 'modeling', 'results', 'predictor_set.pkl'), 'rb') as fp:
            predictors = pickle.load(fp)
