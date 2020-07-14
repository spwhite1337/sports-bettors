import os
import re
import json

import pandas as pd
from tqdm import tqdm

from config import ROOT_DIR, logger


def curate_nfl():
    save_dir = os.path.join(ROOT_DIR, 'data', 'nfl', 'curation')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Remove spaces, dashes from feature labels
    def _clean_label(l):
        return re.sub('\.', '', re.sub(' ', '', re.sub('-', '', l)))

    curation = []
    logger.info('Importing NFL Data to Pandas')
    for team in tqdm([fn for fn in os.listdir(os.path.join(ROOT_DIR, 'data', 'nfl', 'raw')) if '_raw' in fn]):
        with open(os.path.join(ROOT_DIR, 'data', 'nfl', 'raw', team)) as fp:
            team_data = json.load(fp)

        # Iterate through dates for team
        for date, game_data in team_data.items():
            # Save date information of game
            curated = {'year': pd.Timestamp(date).year, 'month': pd.Timestamp(date).month,
                       'day': pd.Timestamp(date).day, 'away_team': game_data['teams'][0],
                       'home_team': game_data['teams'][1]}

            # Wrangle features for home team
            home_features = ['home_' + _clean_label(feature) for feature in game_data['features']]
            home_vals = [val[0] for val in game_data['values']]
            curated.update({label: val for label, val in zip(home_features, home_vals)})

            # Wrangle features for away team
            away_features = ['away_' + _clean_label(feature) for feature in game_data['features']]
            away_vals = [val[1] for val in game_data['values']]
            curated.update({label: val for label, val in zip(away_features, away_vals)})

            # Wrangle home points
            score_labels = ['pts_Q1', 'pts_Q2', 'pts_Q3', 'pts_Q4']
            home_score_labels = ['home_' + label for label in score_labels]
            home_scores = [val for val in game_data['quarter_values'][0][2:6] + [game_data['quarter_values'][0][-1]]]
            curated.update({label: val for label, val in zip(home_score_labels, home_scores)})

            # Wrangle away points
            away_score_labels = ['away_' + label for label in score_labels]
            away_scores = [val for val in game_data['quarter_values'][1][2:6] + [game_data['quarter_values'][1][-1]]]
            curated.update({label: val for label, val in zip(away_score_labels, away_scores)})

            # Gather
            curation.append(curated)
    df_curation = pd.DataFrame.from_records(curation)
    logger.info('Saving Curation with shape: {}'.format(df_curation.shape))
    df_curation.to_csv(os.path.join(save_dir, 'df_curation.csv'), index=False)
