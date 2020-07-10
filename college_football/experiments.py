import os
from modeling.models import FootballBettingAid

from config import ROOT_DIR, logger


def run_experiments():
    df = FootballBettingAid.etl(os.path.join(ROOT_DIR, 'data', 'df_curated.csv'))
    for random_effect in FootballBettingAid.random_effects:
        for response in FootballBettingAid.responses:
            for feature_set in FootballBettingAid.feature_sets.keys():
                logger.info('{} ~ {} | {}'.format(feature_set, response, random_effect))
                aid = FootballBettingAid(random_effect=random_effect, features=feature_set, response=response)
