import os

import pandas as pd

from config import ROOT_DIR, logger


def curate_data():
    """
    Curate downloaded data into a set for modeling / plotting.
    """
    logger.info('Load Raw Data.')
    df_games = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'df_games.csv'))
    df_stats = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'df_stats.csv'))
    df_rankings = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'df_rankings.csv'))

    logger.info('Wrangle stats.')
    for home_away in ['home', 'away']:
        # Pass completion - attempts
        df_stats[home_away + '_passCompletions'] = df_stats[home_away + '_completionAttempts'].\
            apply(lambda pca: int(pca.split('-')[0]) if isinstance(pca, str) else pca)
        df_stats[home_away + '_passAttempts'] = df_stats[home_away + '_completionAttempts'].\
            apply(lambda pca: int(pca.split('-')[1]) if isinstance(pca, str) else pca)

        # Third Downs
        df_stats[home_away + '_thirdDownCompletions'] = df_stats[home_away + '_thirdDownEff'].\
            apply(lambda tde: int(tde.split('-')[0]) if isinstance(tde, str) else tde)
        df_stats[home_away + '_thirdDownAttempts'] = df_stats[home_away + '_thirdDownEff'].\
            apply(lambda tde: int(tde.split('-')[1]) if isinstance(tde, str) else tde)

        # Fourth Downs
        df_stats[home_away + '_fourthDownCompletions'] = df_stats[home_away + '_fourthDownEff'].\
            apply(lambda fde: int(fde.split('-')[0]) if isinstance(fde, str) else fde)
        df_stats[home_away + '_fourthDownAttempts'] = df_stats[home_away + '_fourthDownEff'].\
            apply(lambda fde: int(fde.split('-')[1]) if isinstance(fde, str) else fde)

        # Penalties
        df_stats[home_away + '_numPenalties'] = df_stats[home_away + '_totalPenaltiesYards'].\
            apply(lambda tpy: int(tpy.split('-')[0]) if isinstance(tpy, str) else tpy)
        df_stats[home_away + '_penaltyYards'] = df_stats[home_away + '_totalPenaltiesYards'].\
            apply(lambda tpy: int(tpy.split('-')[1]) if isinstance(tpy, str) else tpy)

        # Possession Time
        df_stats[home_away + '_possessionTime'] = df_stats[home_away + '_possessionTime'].\
            apply(lambda pt: int(pt.split(':')[0]) + int(pt.split(':')[1]) / 60 if isinstance(pt, str) else pt)

        # Fill kicking points with 0
        df_stats[home_away + '_kickingPoints'] = df_stats[home_away + '_kickingPoints'].fillna(0)

    # Drop na
    df_stats = df_stats.dropna(axis=0)
    logger.info('Returned {} games'.format(df_stats.shape[0]))

    logger.info('Matchups and stats')

