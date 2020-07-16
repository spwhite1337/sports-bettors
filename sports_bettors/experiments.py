import os
import argparse

from sports_bettors.utils.college_football.models import CollegeFootballBettingAid
from sports_bettors.utils.nfl.models import NFLBettingAid

from config import Config, logger

betting_aids = {'nfl': NFLBettingAid, 'college_football': CollegeFootballBettingAid}


def execute_experiments(league: str, overwrite: bool = False, debug: bool = False):
    """
    Execute experiments defined from betting aid objects
    """
    betting_aid = betting_aids[league]
    for random_effect in betting_aid.random_effects:
        for feature_set in betting_aid.feature_sets.keys():
            for response in betting_aid.responses:
                # Skip combinations that are over-specified
                if (feature_set == 'PointScored') and (response == 'TotalPoints'):
                    continue
                if debug:
                    if (feature_set != 'RushOnly') or (response not in ['TotalPoints', 'Win']) or \
                            (random_effect != 'team'):
                        continue

                # Check if model already fit
                if not overwrite:
                    if os.path.exists(os.path.join(Config.RESULTS_DIR, league, response, feature_set, random_effect,
                                                   'model_{}.pkl'.format(Config.version))):
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
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    assert args.league in betting_aids.keys()
    logger.info('Running Experiments for {}; Overwrite {}'.format(args.league, args.overwrite))
    execute_experiments(args.league, args.overwrite, args.debug)
