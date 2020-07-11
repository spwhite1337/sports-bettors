from modeling.models import FootballBettingAid

from config import logger


def run_experiments():
    for random_effect in FootballBettingAid.random_effects:
        for response in FootballBettingAid.responses:
            for feature_set in FootballBettingAid.feature_sets.keys():
                logger.info('{} ~ {} | {}'.format(feature_set, response, random_effect))
                aid = FootballBettingAid(random_effect=random_effect, features=feature_set, response=response)
                aid.fit()
                aid.diagnose()
                aid.save()
