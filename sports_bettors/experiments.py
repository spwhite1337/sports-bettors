import os
import pickle
import argparse
from sports_bettors.utils.college_football.models import CollegeFootballBettingAid

from config import ROOT_DIR, logger, version


def run_experiments():
    """
    Generate a series of models with varying inputs / outputs
    """
    parser = argparse.ArgumentParser(prog='Run experiments.')
    parser.add_argument('--league', required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--predictors', action='store_true', help='Generate predictor set from fit models.')
    args = parser.parse_args()

    if args.league == 'sports_bettors':
        predictors = {}
        for random_effect in CollegeFootballBettingAid.random_effects:
            for feature_set in CollegeFootballBettingAid.feature_sets.keys():
                for response in CollegeFootballBettingAid.responses:
                    if not args.overwrite:
                        # Check if model already fit
                        if os.path.exists(os.path.join(ROOT_DIR, 'modeling', 'results', args.league, response,
                                                       feature_set, random_effect,
                                                       'classifier_{}.pkl'.format(version))):
                            logger.info('{} ~ {} | {} already exists, skipping'.format(feature_set, response,
                                                                                       random_effect))
                            continue
                    logger.info('{} ~ {} | {}'.format(feature_set, response, random_effect))
                    aid = CollegeFootballBettingAid(random_effect=random_effect, features=feature_set, response=response)
                    aid.fit()
                    aid.diagnose()
                    aid.save()

    elif args.league == 'nfl':
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    if args.predictors:
        base_dir = os.path.join(ROOT_DIR, 'modeling', 'results')
        for league in os.listdir(base_dir):
            for response in os.listdir(os.path.join(base_dir, league)):
                for feature_set in os.listdir(os.path.join(base_dir, league, response)):
                    for random_effect in os.listdir(os.path.join(base_dir, league, response, feature_set)):
                        with open(os.path.join(base_dir, league, response, feature_set, random_effect,
                                               'classifier_{}.pkl'.format(version)), 'rb') as fp:
                            aid = pickle.load(fp)
                        predictors[(random_effect, feature_set, response)] = aid.predictor
            with open(os.path.join(ROOT_DIR, 'modeling', 'results', league, 'predictor_set_{}.pkl'.format(version)),
                      'wb') as fp:
                pickle.dump(predictors, fp)
