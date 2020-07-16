import os
import pickle
import argparse
from pprint import pformat

from sports_bettors.utils.nfl.models import NFLBettingAid
from sports_bettors.utils.college_football.models import CollegeFootballBettingAid

from config import ROOT_DIR, logger, version


class SportsPredictor(object):
    """
    Object to organize predictions of sporting events conditioned on inputs where applicable
    """
    load_dir = os.path.join(ROOT_DIR, 'modeling', 'results')

    def __init__(self, league: str, random_effect: str, feature_set: str, version: str = version):
        logger.info('Loading predictor set for {}'.format(league))
        with open(os.path.join(self.load_dir, league, 'predictor_set_{}.pkl'.format(version)), 'rb') as fp:
            self.predictors = pickle.load(fp)
        # Config model
        self.league = league
        self.aid = {'nfl': NFLBettingAid, 'college_football': CollegeFootballBettingAid}.get(league)
        self.random_effect = random_effect
        self.feature_set = feature_set

        # Raise errors
        if self.aid is None:
            raise NotImplementedError('{} Not Implemented'.format(league))
        assert random_effect in self.aid.random_effects

    def predict(self, inputs: dict):
        """
        Predict with all models in predictor set, imputing missing values to the mean of the training set
        """
        assert 'RandomEffect' in inputs.keys()

        logger.info(inputs)
        outputs = {}
        for response in self.aid.responses:
            output = self.predictors[(self.random_effect, self.feature_set, response)](inputs.copy())
            outputs[(self.random_effect, self.feature_set, response)] = output

        logger.info(pformat(outputs))


def prediction(league: str, random_effect: str, feature_set: str, inputs: dict):
    predictor = SportsPredictor(league=league, random_effect=random_effect, feature_set=feature_set)
    predictor.predict(inputs)


def prediction_cli():
    parser = argparse.ArgumentParser(prog='Sports Predictions')
    parser.add_argument('--league', type=str, required=True)
    parser.add_argument('--feature_set', type=str, required=True)
    parser.add_argument('--random_effect', type=str, required=True)
    parser.add_argument('--random_effect_val', type=str, required=False)
    args = parser.parse_args()

    # Get aid
    aid = {'nfl': NFLBettingAid, 'college_football': CollegeFootballBettingAid}[args.league]
    if args.feature_set not in aid.feature_sets:
        raise ValueError('feature_set must be in {}'.format(aid.feature_sets))
    if args.random_effect not in aid.random_effects:
        raise ValueError('random_effect must be in {}'.format(aid.random_effects))

    inputs = {'RandomEffect': args.random_effect_val}
    for feature in aid.feature_sets[args.feature_set].features:
        inputs[feature] = float(input('Input Value for {}: '.format(feature)))

    # Get predictions
    prediction(args.league, args.random_effect, args.feature_set, inputs)

