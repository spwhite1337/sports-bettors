import os
import pickle
import argparse

from sports_bettors.utils.college_football.models import CollegeFootballBettingAid
from sports_bettors.utils.nfl.models import NFLBettingAid

from config import ROOT_DIR, logger, version

betting_aids = {'nfl': NFLBettingAid, 'college_football': CollegeFootballBettingAid}


def execute_experiments(league: str, overwrite: bool = False):
    betting_aid = betting_aids[league]
    for random_effect in betting_aid.random_effects:
        for feature_set in betting_aid.feature_sets.keys():
            for response in betting_aid.responses:
                # Check if model already fit
                if not overwrite:
                    if os.path.exists(os.path.join(ROOT_DIR, 'modeling', 'results', league, response,
                                                   feature_set, random_effect,
                                                   'classifier_{}.pkl'.format(version))):
                        logger.info('{} ~ {} | {} already exists, skipping'.format(feature_set, response,
                                                                                   random_effect))
                        continue
                # Fit, Diagnose, and save model
                logger.info('{} ~ {} | {}'.format(feature_set, response, random_effect))
                aid = betting_aid(random_effect=random_effect, features=feature_set, response=response)
                aid.fit()
                aid.diagnose()
                aid.save()


def run_experiments():
    """
    Run experiments for a league
    """
    parser = argparse.ArgumentParser(prog='Run experiments.')
    parser.add_argument('--league', required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--predictors', action='store_true', help='Generate predictor set from fit models.')
    args = parser.parse_args()

    assert args.league in betting_aids.keys()
    logger.info('Running Experiments for {}; Overwrite {}'.format(args.league, args.overwrite))
    execute_experiments(args.league, args.overwrite)

    if args.predictors:
        logger.info('Generating Predictor Sets for {}'.format(args.league))
        predictors = {}
        base_dir = os.path.join(ROOT_DIR, 'modeling', 'results', args.league)
        for response in os.listdir(os.path.join(base_dir, args.league)):
            for feature_set in os.listdir(os.path.join(base_dir, args.league, response)):
                for random_effect in os.listdir(os.path.join(base_dir, args.league, response, feature_set)):
                    with open(os.path.join(base_dir, args.league, response, feature_set, random_effect,
                                           'classifier_{}.pkl'.format(version)), 'rb') as fp:
                        aid = pickle.load(fp)
                    predictors[(random_effect, feature_set, response)] = aid.predictor

        logger.info('Saving Predictor Set for {}'.format(args.league))
        with open(os.path.join(ROOT_DIR, 'modeling', 'results', args.league, 'predictor_set_{}.pkl'.format(version)),
                  'wb') as fp:
            pickle.dump(predictors, fp)
