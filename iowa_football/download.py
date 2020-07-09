import os
import requests
from datetime import datetime
from typing import Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

from config import ROOT_DIR, logger


class DownloadCollegeFootballData(object):
    """
    Object to download data from college football data api
    """
    base_url = 'https://api.collegefootballdata.com/'

    years = [int(2000 + year) for year in range(int(pd.Timestamp(datetime.now()).year - 2000))]

    endpoints = {
        'games':  'games/',
        'stats': 'games/teams'
    }

    def download_games(self) -> pd.DataFrame:
        """
        Download Game Information (no stats) for a year
        """
        logger.info('Downloading Game data.')
        df = []
        for year in tqdm(self.years):
            # Return data
            r = requests.get(self.base_url + self.endpoints['games'], params={'year': year})

            # Convert to dataframe
            df_year = pd.DataFrame.from_records(r.json())

            # Only retain useful fields, rename as necessary
            df_year = df_year[
                ['id', 'season', 'week', 'season_type', 'home_team', 'away_team', 'home_points', 'away_points']
            ].rename(columns={'id': 'game_id'})

            # Gather
            df.append(df_year)
        df = pd.concat(df).reset_index(drop=True)

        # Save
        logger.info('Saving games data.')
        df.to_csv(os.path.join(ROOT_DIR, 'data', 'df_games.csv'), index=False)

        return df

    def download_stats(self, df_games: pd.DataFrame = None) -> Tuple[pd.DataFrame, list]:
        """
        Download stats for each game
        """
        if df_games is None:
            df_games = self.download_games()
        assert 'game_id' in df_games.columns

        logger.info('Downloading Stats.')
        df_stats, df_fails = [], []
        for game_id in tqdm(set(df_games['game_id'])):
            params = {'gameId': game_id}

            # Try a download, catch connection failures
            try:
                r = requests.get(self.base_url + self.endpoints['stats'], params=params)
            except Exception as err:
                logger.info('{}, {}'.format(game_id, err))
                df_fail = pd.DataFrame({'game_id': game_id, 'error': err}, index=[0])
                df_fails.append(df_fail)
                continue

            if r.status_code != 200:
                df_fail = pd.DataFrame({'game_id': game_id, 'error': r.status_code}, index=[0])
                df_fails.append(df_fail)
                continue

            if len(r.json()) == 0:
                df_fail = pd.DataFrame({'game_id': game_id, 'error': 'No Response'}, index=[0])
                df_fails.append(df_fail)
                continue

            # Get list of stats where each entry is a team
            stats_by_team = r.json()[0]['teams']
            df_game = []
            for team in stats_by_team:
                df_game_long = pd.DataFrame.from_records(team['stats'])
                df_game_long['category'] = team['homeAway'] + '_' + df_game_long['category']
                df_game.append(df_game_long)
            df_game = pd.concat(df_game)
            # Pivot
            df_game = pd.pivot_table(df_game, columns='category', values='stat', aggfunc='first').reset_index(drop=True)
            df_stats.append(df_game.assign(game_id=game_id))

        df_stats = pd.concat(df_stats).reset_index(drop=True)
        df_fails = pd.concat(df_fails).reset_index(drop=True)

        # Saving stats
        logger.info('Saving Stats.')
        df_stats.to_csv(os.path.join(ROOT_DIR, 'data', 'df_stats.csv'), index=False)
        df_fails.to_csv(os.path.join(ROOT_DIR, 'data', 'df_failed_stats.csv'), index=False)

        return df_stats, df_fails


def download():
    downloader = DownloadCollegeFootballData()
    downloader.download_stats()
