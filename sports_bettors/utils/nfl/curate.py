import os
import re
import json

import pandas as pd
import numpy as np
from tqdm import tqdm

from config import Config, logger


def curate_nfl():
    save_dir = os.path.join(Config.CURATED_DIR, 'nfl')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Remove spaces, dashes from feature labels
    def _clean_label(l):
        return re.sub('\.', '', re.sub(' ', '', re.sub('-', '', l)))

    curation = []
    logger.info('Importing NFL Data to Pandas')
    for team in tqdm([fn for fn in os.listdir(os.path.join(Config.RAW_DIR, 'nfl')) if '_raw' in fn]):
        with open(os.path.join(Config.RAW_DIR, 'nfl', team)) as fp:
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
            score_labels = ['pts_Q1', 'pts_Q2', 'pts_Q3', 'pts_Q4', 'points']
            home_score_labels = ['home_' + label for label in score_labels]
            home_scores = [val for val in game_data['quarter_values'][0][2:6] + [game_data['quarter_values'][0][-1]]]
            curated.update({label: val for label, val in zip(home_score_labels, home_scores)})

            # Wrangle away points
            away_score_labels = ['away_' + label for label in score_labels]
            away_scores = [val for val in game_data['quarter_values'][1][2:6] + [game_data['quarter_values'][1][-1]]]
            curated.update({label: val for label, val in zip(away_score_labels, away_scores)})

            # Gather
            curation.append(curated)
    df_curation = pd.DataFrame.from_records(curation).drop_duplicates()
    logger.info('Saving Curation with shape: {}'.format(df_curation.shape))
    df_curation.to_csv(os.path.join(save_dir, 'df_stats.csv'), index=False)

    # Clean up fields
    def _dash_curate(stat: str, idx: int):
        stat = re.sub('--', '-', stat)
        try:
            return int(float(stat.split('-')[idx])) if len(stat) > 2 else np.nan
        except Exception as err:
            logger.info('{}: {}'.format(stat, err))
            return np.nan

    for home_away in ['home', 'away']:
        # Clean up passing stats
        df_curation[home_away + '_passCompletions'] = df_curation[home_away + '_CmpAttYdTDINT'].apply(
            lambda stat: _dash_curate(stat, 0))
        df_curation[home_away + '_passAttempts'] = df_curation[home_away + '_CmpAttYdTDINT'].apply(
            lambda stat: _dash_curate(stat, 1))
        df_curation[home_away + '_passYards'] = df_curation[home_away + '_CmpAttYdTDINT'].apply(
            lambda stat: _dash_curate(stat, 2))
        df_curation[home_away + '_passTDs'] = df_curation[home_away + '_CmpAttYdTDINT'].apply(
            lambda stat: _dash_curate(stat, 3))
        df_curation[home_away + '_interceptions'] = df_curation[home_away + '_CmpAttYdTDINT'].apply(
            lambda stat: _dash_curate(stat, 4))
        df_curation = df_curation.drop(home_away + '_CmpAttYdTDINT', axis=1)

        # Clean up rushing stats
        df_curation[home_away + '_rushAttempts'] = df_curation[home_away + '_RushYdsTDs'].apply(
            lambda stat: _dash_curate(stat, 0))
        df_curation[home_away + '_rushYards'] = df_curation[home_away + '_RushYdsTDs'].apply(
            lambda stat: _dash_curate(stat, 1))
        df_curation[home_away + '_rushTDs'] = df_curation[home_away + '_RushYdsTDs'].apply(
            lambda stat: _dash_curate(stat, 2))
        df_curation = df_curation.drop(home_away + '_RushYdsTDs', axis=1)

        # Fourth Down
        df_curation[home_away + '_FourthDownConv'] = df_curation[home_away + '_FourthDownConv'].fillna('0-0')
        df_curation[home_away + '_fourthDownConversions'] = df_curation[home_away + '_FourthDownConv'].apply(
            lambda stat: _dash_curate(stat, 0))
        df_curation[home_away + '_fourthDownAttempts'] = df_curation[home_away + '_FourthDownConv'].apply(
            lambda stat: _dash_curate(stat, 1))
        df_curation = df_curation.drop(home_away + '_FourthDownConv', axis=1)

        # Fumbles Lost
        df_curation[home_away + '_fumbles'] = df_curation[home_away + '_FumblesLost'].apply(
            lambda stat: _dash_curate(stat, 0))
        df_curation[home_away + '_fumblesLost'] = df_curation[home_away + '_FumblesLost'].apply(
            lambda stat: _dash_curate(stat, 1))
        df_curation = df_curation.drop(home_away + '_FumblesLost', axis=1)

        # Clean up penalties
        df_curation[home_away + '_penalties'] = df_curation[home_away + '_PenaltiesYards'].apply(
            lambda stat: _dash_curate(stat, 0))
        df_curation[home_away + '_penaltyYards'] = df_curation[home_away + '_PenaltiesYards'].apply(
            lambda stat: _dash_curate(stat, 1))
        df_curation = df_curation.drop(home_away + '_PenaltiesYards', axis=1)

        # Clean up sacks
        df_curation[home_away + '_sacks'] = df_curation[home_away + '_SackedYards'].apply(
            lambda stat: _dash_curate(stat, 0))
        df_curation[home_away + '_sackedYards'] = df_curation[home_away + '_SackedYards'].apply(
            lambda stat: _dash_curate(stat, 1))
        df_curation = df_curation.drop(home_away + '_SackedYards', axis=1)

        # Third Down
        df_curation[home_away + '_ThirdDownConv'] = df_curation[home_away + '_ThirdDownConv'].fillna('0-0')
        df_curation[home_away + '_thirdDownConversions'] = df_curation[home_away + '_ThirdDownConv'].apply(
            lambda stat: _dash_curate(stat, 0))
        df_curation[home_away + '_thirdDownAttempts'] = df_curation[home_away + '_ThirdDownConv'].apply(
            lambda stat: _dash_curate(stat, 1))
        df_curation = df_curation.drop(home_away + '_ThirdDownConv', axis=1)

        # Possession Time
        df_curation[home_away + '_TimeofPossession'] = df_curation[home_away + '_TimeofPossession'].fillna('00:00')
        df_curation[home_away + '_possessionTime'] = df_curation[home_away + '_TimeofPossession'].apply(
            lambda stat: float(stat.split(':')[0]) + float(stat.split(':')[1]) / 60 if len(stat) > 4 else np.nan)
        df_curation = df_curation.drop(home_away + '_TimeofPossession', axis=1)

        # Convert back to NA
        df_curation[home_away + '_possessionTime'] = df_curation.apply(
            lambda row: row[home_away + '_possessionTime'] if row['year'] > 1983 else np.nan, axis=1
        )
        df_curation[home_away + '_fourthDownConversions'] = df_curation.apply(
            lambda row: row[home_away + '_fourthDownConversions'] if row['year'] > 1991 else np.nan, axis=1
        )
        df_curation[home_away + '_fourthDownAttempts'] = df_curation.apply(
            lambda row: row[home_away + '_fourthDownAttempts'] if row['year'] > 1991 else np.nan, axis=1
        )
        df_curation[home_away + '_thirdDownConversions'] = df_curation.apply(
            lambda row: row[home_away + '_thirdDownConversions'] if row['year'] > 1991 else np.nan, axis=1
        )
        df_curation[home_away + '_thirdDownAttempts'] = df_curation.apply(
            lambda row: row[home_away + '_thirdDownAttempts'] if row['year'] > 1991 else np.nan, axis=1
        )

    # Wrangle from home / away to team / opponent
    df_modeling = []
    all_teams = set(list(df_curation['home_team']) + list(df_curation['away_team']))
    for team in tqdm(all_teams):
        # Games where team is home
        df_home = df_curation[df_curation['home_team'] == team].copy()
        df_home['team'] = team
        df_home['is_home'] = 1
        df_home['opponent'] = df_home['away_team']
        df_home = df_home.drop(['home_team', 'away_team'], axis=1)
        df_home.columns = [re.sub('away_', 'opp_', re.sub('home_', '', col)) for col in df_home.columns]
        df_modeling.append(df_home)

        # Games where team is away
        df_away = df_curation[df_curation['away_team'] == team].copy()
        df_away['team'] = team
        df_away['is_home'] = 0
        df_away['opponent'] = df_away['home_team']
        df_away = df_away.drop(['home_team', 'away_team'], axis=1)
        df_away.columns = [re.sub('home_', 'opp_', re.sub('away_', '', col)) for col in df_away.columns]
        df_modeling.append(df_away)
    df_modeling = pd.concat(df_modeling, sort=True).drop_duplicates().reset_index(drop=True)

    # Matchup
    def _define_matchup(main_team, opponent):
        return '_vs_'.join(sorted([main_team, opponent]))
    df_modeling['matchup'] = df_modeling.apply(lambda row: _define_matchup(row['team'], row['opponent']), axis=1)

    logger.info('Save Curated data for {} games.'.format(df_modeling.shape[0]))
    df_modeling.to_csv(os.path.join(Config.CURATED_DIR, 'nfl', 'df_curated.csv'), index=False)
