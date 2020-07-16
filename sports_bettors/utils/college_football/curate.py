import os
import re

from tqdm import tqdm
import pandas as pd

from config import Config, logger


def curate_college():
    """
    Curate sports_bettors football data
    """
    RAW_DIR = os.path.join(Config.RAW_DIR, 'college_football')
    CURATION_DIR = os.path.join(Config.CURATED_DIR, 'college_football')
    if not os.path.exists(CURATION_DIR):
        os.makedirs(CURATION_DIR)

    logger.info('Load Raw Data.')
    df_games = pd.read_csv(os.path.join(RAW_DIR, 'df_games.csv'))
    df_stats = pd.read_csv(os.path.join(RAW_DIR, 'df_stats.csv'))
    df_rankings = pd.read_csv(os.path.join(RAW_DIR, 'df_rankings.csv'))

    logger.info('Wrangle stats.')
    # Drop data errors
    df_stats = df_stats[df_stats['away_totalPenaltiesYards'] != '7--4953']

    for home_away in ['home', 'away']:
        # Pass completion - attempts
        df_stats[home_away + '_passCompletions'] = df_stats[home_away + '_completionAttempts']. \
            apply(lambda pca: int(pca.split('-')[0]) if isinstance(pca, str) else pca)
        df_stats[home_away + '_passAttempts'] = df_stats[home_away + '_completionAttempts']. \
            apply(lambda pca: int(pca.split('-')[1]) if isinstance(pca, str) else pca)
        df_stats = df_stats.drop(home_away + '_completionAttempts', axis=1)

        # Third Downs
        df_stats[home_away + '_thirdDownCompletions'] = df_stats[home_away + '_thirdDownEff']. \
            apply(lambda tde: int(tde.split('-')[0]) if isinstance(tde, str) else tde)
        df_stats[home_away + '_thirdDownAttempts'] = df_stats[home_away + '_thirdDownEff']. \
            apply(lambda tde: int(tde.split('-')[1]) if isinstance(tde, str) else tde)
        df_stats = df_stats.drop(home_away + '_thirdDownEff', axis=1)

        # Fourth Downs
        df_stats[home_away + '_fourthDownCompletions'] = df_stats[home_away + '_fourthDownEff']. \
            apply(lambda fde: int(fde.split('-')[0]) if isinstance(fde, str) else fde)
        df_stats[home_away + '_fourthDownAttempts'] = df_stats[home_away + '_fourthDownEff']. \
            apply(lambda fde: int(fde.split('-')[1]) if isinstance(fde, str) else fde)
        df_stats = df_stats.drop(home_away + '_fourthDownEff', axis=1)

        # Penalties
        df_stats[home_away + '_numPenalties'] = df_stats[home_away + '_totalPenaltiesYards']. \
            apply(lambda tpy: int(tpy.split('-')[0]) if isinstance(tpy, str) else tpy)
        df_stats[home_away + '_penaltyYards'] = df_stats[home_away + '_totalPenaltiesYards']. \
            apply(lambda tpy: int(tpy.split('-')[1]) if isinstance(tpy, str) else tpy)
        df_stats = df_stats.drop(home_away + '_totalPenaltiesYards', axis=1)

        # Possession Time
        df_stats[home_away + '_possessionTime'] = df_stats[home_away + '_possessionTime']. \
            apply(lambda pt: int(pt.split(':')[0]) + int(pt.split(':')[1]) / 60 if isinstance(pt, str) else pt)

        # Fill kicking points with 0
        df_stats[home_away + '_kickingPoints'] = df_stats[home_away + '_kickingPoints'].fillna(0)

    # Drop na
    df_stats = df_stats.dropna(axis=0)
    logger.info('Returned {} games'.format(df_stats.shape[0]))

    logger.info('Merge Match-ups and stats')
    df_stats = df_stats.merge(df_games, on='game_id', how='inner')

    logger.info('Merge Rankings and Stats')
    for poll, df_poll in df_rankings.groupby('poll'):
        rank_col_name = re.sub(' ', '', poll) + 'Rank'
        df_poll_sub = df_poll. \
            rename(columns={'rank': rank_col_name, 'year': 'season'})

        # Merge with stats
        for home_away in ['home', 'away']:
            df_poll_sub[home_away + '_team'] = df_poll_sub['school']
            df_poll_sub[home_away + '_' + rank_col_name] = df_poll_sub[rank_col_name]
            df_stats = df_stats.merge(df_poll_sub[
                                          ['season', 'week', home_away + '_team', home_away + '_' + rank_col_name]
                                      ],
                                      on=['season', 'week', home_away + '_team'],
                                      how='left')
    # Clean up team names
    df_stats['home_team'] = df_stats['home_team'].apply(lambda team: re.sub(' ', '', team))
    df_stats['away_team'] = df_stats['away_team'].apply(lambda team: re.sub(' ', '', team))

    logger.info('Define Modeling Dataset.')
    df_modeling = []
    all_teams = set(list(df_stats['home_team']) + list(df_stats['away_team']))
    for team in tqdm(all_teams):
        # Games where team is home
        df_home = df_stats[df_stats['home_team'] == team].copy()
        df_home['team'] = team
        df_home['is_home'] = 1
        df_home['opponent'] = df_home['away_team']
        df_home = df_home.drop(['home_team', 'away_team'], axis=1)
        df_home.columns = [re.sub('away_', 'opp_', re.sub('home_', '', col)) for col in df_home.columns]
        df_modeling.append(df_home)

        # Games where team is away
        df_away = df_stats[df_stats['away_team'] == team].copy()
        df_away['team'] = team
        df_away['is_home'] = 0
        df_away['opponent'] = df_away['home_team']
        df_away = df_away.drop(['home_team', 'away_team'], axis=1)
        df_away.columns = [re.sub('home_', 'opp_', re.sub('away_', '', col)) for col in df_away.columns]
        df_modeling.append(df_away)
    df_modeling = pd.concat(df_modeling, sort=True).reset_index(drop=True)

    # Matchup
    def _define_matchup(main_team, opponent):
        return '_vs_'.join(sorted([main_team, opponent]))
    df_modeling['matchup'] = df_modeling.apply(lambda row: _define_matchup(row['team'], row['opponent']), axis=1)

    logger.info('Save Curated data for {} games.'.format(df_modeling.shape))
    df_modeling.to_csv(os.path.join(CURATION_DIR, 'df_curated.csv'), index=False)
