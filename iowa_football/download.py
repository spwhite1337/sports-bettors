import os
import requests
from datetime import datetime
from typing import Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

from config import logger


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

        return df

    def download_stats(self, df_games: pd.DataFrame = None) -> Tuple[pd.DataFrame, list]:
        """
        Download stats for each game
        """
        if df_games is None:
            df_games = self.download_games()
        assert 'game_id' in df_games.columns

        df_stats, didnt_work = [], []
        for game_id in set(df_games['game_id']):
            params = {'gameId': game_id}
            try:
                r = requests.get(self.base_url + self.endpoints['stats'], params=params)
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
            except Exception as err:
                logger.info(err)
                didnt_work.append(game_id)
        df_stats = pd.concat(df_stats).reset_index(drop=True)

        return df_stats, didnt_work
