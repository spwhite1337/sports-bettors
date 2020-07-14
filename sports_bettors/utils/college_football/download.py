import os
import requests
from datetime import datetime
from typing import Tuple

import pandas as pd
from tqdm import tqdm

from config import ROOT_DIR, logger


class DownloadCollegeFootballData(object):
    """
    Object to download data from sports_bettors football data api
    """
    base_url = 'https://api.collegefootballdata.com/'

    years = [int(2000 + year) for year in range(int(pd.Timestamp(datetime.now()).year - 2000))]

    # Stats to retain as features
    features = [
        'yardsPerRushAttempt',
        'yardsPerPass',
        'turnovers',
        'totalYards',
        'totalPenaltiesYards',
        'thirdDownEff',
        'rushingYards',
        'rushingTDs',
        'rushingAttempts',
        'possessionTime',
        'passingTDs',
        'netPassingYards',
        'interceptions',
        'fumblesRecovered',
        'fumblesLost',
        'fourthDownEff',
        'firstDowns',
        'completionAttempts',
        'kickingPoints'
    ]

    def __init__(self, save_dir: str = None):
        self.save_dir = os.path.join(ROOT_DIR, 'data', 'college_football', 'raw') if save_dir is None else save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def download_games(self) -> pd.DataFrame:
        """
        Download Game Information (no stats) for a year
        """
        logger.info('Downloading Game data.')
        df = []
        for year in tqdm(self.years):
            # Return data
            r = requests.get(self.base_url + 'games/', params={'year': year})

            # Convert to dataframe
            df_year = pd.DataFrame.from_records(r.json())

            # Only retain useful fields, rename as necessary
            df_year = df_year[
                ['id', 'season', 'week', 'season_type', 'home_team', 'away_team', 'home_points', 'away_points']
            ].rename(columns={'id': 'game_id'})

            # Gather
            df.append(df_year)
        df = pd.concat(df, sort=True).reset_index(drop=True)

        # Save
        logger.info('Saving games data.')
        df.to_csv(os.path.join(self.save_dir, 'df_games.csv'), index=False)

        return df

    def download_rankings(self) -> Tuple[pd.DataFrame, list]:
        """
        Download team rankings by week based on various polls
        """
        logger.info('Downloading Rankings.')
        df, df_fails = [], []
        for year in tqdm(self.years):

            # Try a download, catch connection failures
            try:
                params = {'year': year}
                r = requests.get(self.base_url + 'rankings', params=params)
            except Exception as err:
                logger.info('{}, {}'.format(year, err))
                df_fail = pd.DataFrame({'year': year, 'error': err}, index=[0])
                df_fails.append(df_fail)
                continue

            if r.status_code != 200:
                logger.info('{}, {}'.format(year, r.status_code))
                df_fail = pd.DataFrame({'year': year, 'error': r.status_code}, index=[0])
                df_fails.append(df_fail)
                continue

            if len(r.json()) == 0:
                df_fail = pd.DataFrame({'year': year, 'error': 'No Data'}, index=[0])
                df_fails.append(df_fail)
                continue

            for week_record in r.json():
                week = week_record['week']
                for poll in week_record['polls']:
                    poll_name = poll['poll']
                    # Only use polls you've heard of
                    if poll_name not in ['AP Top 25', 'BCS Standings', 'Coaches Poll']:
                        continue
                    df_ranks = pd.DataFrame.from_records(poll['ranks']).assign(poll=poll_name, week=week, year=year)
                    # Subset
                    df_ranks = df_ranks[['year', 'week', 'poll', 'rank', 'school', 'conference']]

                    df.append(df_ranks)
        df = pd.concat(df, sort=True).reset_index(drop=True)
        df_fails = pd.concat(df_fails, sort=True).reset_index(drop=True) if len(df_fails) > 0 else pd.DataFrame()

        # Save
        logger.info('Saving Rankings.')
        df.to_csv(os.path.join(ROOT_DIR, 'data', 'sports_bettors', 'df_rankings.csv'), index=False)
        df_fails.to_csv(os.path.join(self.save_dir, 'df_failed_rankings.csv'), index=False)

        return df, df_fails

    def download_stats(self, df_games: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
                r = requests.get(self.base_url + 'games/teams', params=params)
            except Exception as err:
                logger.info('{}, {}'.format(game_id, err))
                df_fail = pd.DataFrame({'game_id': game_id, 'error': err}, index=[0])
                df_fails.append(df_fail)
                continue

            if r.status_code != 200:
                logger.info('{}, {}'.format(game_id, r.status_code))
                df_fail = pd.DataFrame({'game_id': game_id, 'error': r.status_code}, index=[0])
                df_fails.append(df_fail)
                continue

            if len(r.json()) == 0:
                df_fail = pd.DataFrame({'game_id': game_id, 'error': 'No Data'}, index=[0])
                df_fails.append(df_fail)
                continue

            # Get list of stats where each entry is a team
            stats_by_team = r.json()[0]['teams']
            df_game = []
            for team in stats_by_team:
                df_game_long = pd.DataFrame.from_records(team['stats'])

                # Subset for useful stats
                df_game_long = df_game_long[df_game_long['category'].isin(self.features)]

                # Append home/away to category
                df_game_long['category'] = team['homeAway'] + '_' + df_game_long['category']
                df_game.append(df_game_long)
            df_game = pd.concat(df_game, sort=True)

            # Pivot
            df_game = pd.pivot_table(df_game, columns='category', values='stat', aggfunc='first').reset_index(drop=True)
            df_stats.append(df_game.assign(game_id=game_id))

        df_stats = pd.concat(df_stats, sort=True).reset_index(drop=True)
        df_fails = pd.concat(df_fails, sort=True).reset_index(drop=True) if len(df_fails) > 0 else pd.DataFrame()

        logger.info('Downloaded Stats for {} games.'.format(df_stats.shape[0]))
        logger.info('Failed downloads for {} games.'.format(df_fails.shape[0]))

        # Saving stats
        logger.info('Saving Stats.')
        df_stats.to_csv(os.path.join(self.save_dir, 'df_stats.csv'), index=False)
        df_fails.to_csv(os.path.join(self.save_dir, 'df_failed_stats.csv'), index=False)

        return df_stats, df_fails

    def retry_stats(self):
        """
        Some games fail due to connection issues with the API; retry these games here
        """
        failed_ids = os.path.join(ROOT_DIR, 'data', 'sports_bettors', 'df_failed_stats.csv')
        if not os.path.exists(failed_ids):
            logger.info('Download stats before retrying.')
            return
        df = pd.read_csv(failed_ids)

        # Get games with previously downloaded stats
        original_stats = os.path.join(self.save_dir, 'df_stats.csv')
        df_original = pd.read_csv(original_stats)

        # Retry failed games
        df_stats, df_fails = self.download_stats(df)

        # Append
        df_retried = df_original.append(df_stats).drop_duplicates().reset_index(drop=True)

        # Save
        df_retried.to_csv(original_stats, index=False)
        df_fails.to_csv(failed_ids, index=False)
