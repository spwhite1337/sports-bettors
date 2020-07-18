import os
import pickle
import argparse
import pprint
from typing import Tuple

from sports_bettors.utils.nfl.models import NFLBettingAid
from sports_bettors.utils.college_football.models import CollegeFootballBettingAid
from sports_bettors.base import BetPredictor, BaseBettingAid

from config import Config, logger


class SportsPredictor(object):
    """
    Object to organize predictions of sporting events conditioned on inputs where applicable
    """
    load_dir = Config.RESULTS_DIR
    aids = {'nfl': NFLBettingAid, 'college_football': CollegeFootballBettingAid}

    def __init__(self, league: str, version: str = Config.version):
        self.league = league
        self.predictors = None
        self.version = version

    def load(self):
        """
        Load predictor set
        """
        logger.info('Loading predictor set for {}'.format(self.league))
        with open(os.path.join(self.load_dir, self.league, 'predictor_set_{}.pkl'.format(self.version)), 'rb') as fp:
            self.predictors = pickle.load(fp)

    def predict(self, random_effect: str, feature_set: str, inputs: dict) -> dict:
        """
        Predict with all models in predictor set, imputing missing values to the mean of the training set
        """
        aid = self.aids.get(self.league)

        # Raise errors
        if aid is None:
            raise NotImplementedError('{} Not Implemented'.format(self.league))
        assert random_effect in aid.random_effects
        assert 'RandomEffect' in inputs.keys()

        logger.info('Input: {}'.format(inputs))
        outputs = {}
        for response in aid.responses:
            key = (random_effect, feature_set, response)
            if key in self.predictors.keys():
                output = self.predictors[key](inputs)
                outputs[(random_effect, feature_set, response)] = output

        return outputs

    @staticmethod
    def _get_calculator(aid: BaseBettingAid) -> Tuple[dict, Tuple[float, float]]:
        """
        Use the summary of a fit betting aid to create a small predictor dictionary
        """
        # Random effects
        df_re = aid.summary[aid.summary['labels'].str.startswith('a[')].reset_index(drop=True)
        df_re['labels'] = df_re['labels'].map(aid.random_effect_inv)

        # Coefficients
        df_coefs = aid.summary[aid.summary['labels'].str.contains('^b[0-9]', regex=True)].\
            assign(labels=aid.features)

        # Global intercept
        intercept = aid.summary[aid.summary['labels'] == 'mu_a']['mean'].iloc[0]
        intercept_sd = aid.summary[aid.summary['labels'] == 'mu_a']['sd'].iloc[0]

        # Noise
        noise = aid.summary[aid.summary['labels'] == 'sigma_y']['mean'].iloc[0]
        noise_sd = aid.summary[aid.summary['labels'] == 'sigma_y']['sd'].iloc[0]

        # Convert to lightweight predictor
        calculator = {
            'random_effect': dict(zip(
                df_re['labels'],
                list(zip(df_re['mean'] - df_re['sd'], df_re['mean'], df_re['mean'] + df_re['sd']))
            )),
            'coefficients': dict(zip(
                df_coefs['labels'],
                list(zip(df_coefs['mean'] - df_coefs['sd'], df_coefs['mean'], df_coefs['mean'] + df_coefs['sd']))
            ))
        }

        # Add noise if continuous response
        if aid.response_distributions[aid.response] != 'bernoulli_logit':
            calculator['noise'] = (noise - noise_sd, noise, noise + noise_sd)

        return calculator, (intercept, intercept_sd)

    def generate_predictor_set(self):
        """
        Generate predict sets for each league from all the previously fit models
        """
        logger.info('Generating Predictor Sets for {}'.format(self.league))
        predictors = {}
        base_dir = os.path.join(Config.RESULTS_DIR, self.league)
        for response in os.listdir(os.path.join(base_dir)):
            for feature_set in os.listdir(os.path.join(base_dir, response)):
                for random_effect in os.listdir(os.path.join(base_dir, response, feature_set)):
                    # Load predictor
                    aid_path = os.path.join(base_dir, response, feature_set, random_effect,
                                            'aid_{}.pkl'.format(Config.version))
                    with open(aid_path, 'rb') as fp:
                        aid = pickle.load(fp)
                    calculator, re_params = self._get_calculator(aid)
                    predictor = BetPredictor(scales=aid.scales, calculator=calculator, re_params=re_params)
                    predictors[(random_effect, feature_set, response)] = predictor

        logger.info('Saving Predictor Set for {}'.format(self.league))
        with open(os.path.join(Config.RESULTS_DIR, self.league, 'predictor_set_{}.pkl'.format(Config.version)),
                  'wb') as fp:
            pickle.dump(predictors, fp)
        self.predictors = predictors


def api(league: str, random_effect: str, feature_set: str, inputs: dict, display_output: bool = False):
    predictor = SportsPredictor(league=league)
    predictor.load()
    output = predictor.predict(inputs=inputs, random_effect=random_effect, feature_set=feature_set)
    if display_output:
        pp = pprint.PrettyPrinter(indent=4, compact=True)
        logger.info('Output: {}'.format(pp.pprint(output)))
    return output


def api_cli():
    parser = argparse.ArgumentParser(prog='Sports Predictions')
    parser.add_argument('--league', type=str, required=True)
    parser.add_argument('--feature_set', type=str, required=True)
    parser.add_argument('--random_effect', type=str, required=True)
    parser.add_argument('--display_output', action='store_true')
    args = parser.parse_args()

    # Get aid
    aid = {'nfl': NFLBettingAid, 'college_football': CollegeFootballBettingAid}[args.league]
    if args.feature_set not in aid.feature_sets:
        raise ValueError('feature_set must be in {}'.format(aid.feature_sets))
    if args.random_effect not in aid.random_effects:
        raise ValueError('random_effect must be in {}'.format(aid.random_effects))

    # Inputs
    inputs = {'RandomEffect': input('Input Value for RandomEffect ({}): '.format(args.random_effect))}
    for feature in aid.feature_sets[args.feature_set].features:
        inputs[feature] = float(input('Input Value for {}: '.format(feature)))

    # Get predictions
    api(args.league, args.random_effect, args.feature_set, inputs, args.display_output)


def create_predictor_sets():
    parser = argparse.ArgumentParser(prog='Generator Predictor Sets')
    parser.add_argument('--league', required=True)
    args = parser.parse_args()

    # Generator predictor set
    predictor = SportsPredictor(league=args.league)
    predictor.generate_predictor_set()
