import os
import pickle
import argparse
from modeling.models import CollegeFootballBettingAid

from config import ROOT_DIR, logger, version


def run_experiments():
    """
    Generate a series of models with varying inputs / outputs
    """
    parser = argparse.ArgumentParser(prog='Run experiments.')
    parser.add_argument('--league', default='college')
    args = parser.parse_args()

    if args.league == 'college':
        predictors = {}
        for random_effect in CollegeFootballBettingAid.random_effects:
            for feature_set in CollegeFootballBettingAid.feature_sets.keys():
                for response in CollegeFootballBettingAid.responses:
                    logger.info('{} ~ {} | {}'.format(feature_set, response, random_effect))
                    aid = CollegeFootballBettingAid(random_effect=random_effect, features=feature_set, response=response)
                    aid.fit()
                    aid.diagnose()
                    aid.save()

                    # Save predictors
                    predictors[(random_effect, feature_set, response)] = aid.predictor

        with open(os.path.join(ROOT_DIR, 'modeling', 'results', 'college_football',
                               'predictor_set_{}.pkl'.format(version)), 'wb') as fp:
            pickle.dump(predictors, fp)
