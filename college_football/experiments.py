import os
import pickle
from modeling.models import FootballBettingAid

from config import ROOT_DIR, logger


def run_experiments():
    predictors = {}
    for random_effect in FootballBettingAid.random_effects:
        for response in FootballBettingAid.responses:
            for feature_set in FootballBettingAid.feature_sets.keys():
                logger.info('{} ~ {} | {}'.format(feature_set, response, random_effect))
                aid = FootballBettingAid(random_effect=random_effect, features=feature_set, response=response)
                aid.fit()
                aid.diagnose()
                aid.save()

                # Save predictors
                predictors[(random_effect, feature_set, response)] = aid.predictor

    with open(os.path.join(ROOT_DIR, 'modeling', 'results', 'predictor_set.pkl'), 'wb') as fp:
        pickle.dump(predictors, fp)
