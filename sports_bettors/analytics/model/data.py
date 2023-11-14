import re
import os
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
import time
import cfbd

from sports_bettors.analytics.eda.eda import Eda
from config import logger


class Data(Eda):
    training_years = 5
    # https://github.com/nflverse/nfldata
    link_to_data = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
    window = 365

    college_conferences = ['ACC', 'B12', 'B1G', 'SEC', 'Pac-10', 'PAC',
                           # 'Ind'
                           ]

    def __init__(self, league: str = 'nfl', overwrite: bool = False):
        super().__init__()
        self.training_start = datetime.datetime.strftime(
            datetime.datetime.today() - datetime.timedelta(days=self.training_years * 365),
            '%Y-%m-%d',
        )
        self.overwrite = overwrite
        assert league in ['nfl', 'college_football']
        self.league = league
        self.cache_dir = os.path.join(os.getcwd(), 'data', 'sports_bettors', 'cache', self.league)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    @staticmethod
    def _impute_money_line_from_spread(spread: float) -> Optional[float]:
        if spread is None:
            return None
        # Empirically fit from non-imputed data to payout
        p = [0.0525602, -0.08536405]
        payout = 10 ** (p[1] + p[0] * spread)
        # Convert to moneyline
        if payout > 1:
            return payout * 100
        else:
            return -1 / payout * 100

    @staticmethod
    def _parse_college_football(df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _download_college_football(self, predict: bool = False) -> pd.DataFrame:
        """
        Pull data from https://github.com/CFBD/cfbd-python
        As of 10/2023 it is "free to use without restrictions"
        """
        current_year = datetime.datetime.today().year
        if not predict:
            years = list(np.linspace(current_year - self.training_years - 1, current_year, self.training_years + 2))
        else:
            years = list(np.linspace(current_year - 1, current_year, 2))
        season_type = 'regular'
        df, df_raw = [], None
        for year in tqdm(years):
            for conference in tqdm(self.college_conferences):
                if df_raw is not None:
                    continue
                # Rest a bit for the API because it is free
                time.sleep(2)
                try:
                    configuration = cfbd.Configuration()
                    configuration.api_key['Authorization'] = os.environ['API_KEY_COLLEGE_API']
                    configuration.api_key_prefix['Authorization'] = 'Bearer'
                    api_instance = cfbd.BettingApi(cfbd.ApiClient(configuration))
                    api_response = api_instance.get_lines(year=year, season_type=season_type, conference=conference)
                except:
                    logger.error('API Miss')
                    if predict:
                        return pd.read_csv(os.path.join(self.cache_dir, 'df_training.csv'), parse_dates=['gameday'])
                    else:
                        df_raw = pd.read_csv(os.path.join(self.cache_dir, 'df_training_raw.csv'), parse_dates=['gameday'])
                        api_response = []
                records = []
                for b in api_response:
                    record = {
                        'gameday': b.start_date,
                        'game_id': str(year) + '_' + re.sub(' ', '', b.away_team) + '_' + re.sub(' ', '', b.home_team),
                        'away_conference': b.away_conference,
                        'away_team': b.away_team,
                        'away_score': b.away_score,
                        'home_conference': b.home_conference,
                        'home_team': b.home_team,
                        'home_score': b.home_score
                    }
                    for line in b.lines:
                        record['away_moneyline'] = line.away_moneyline
                        record['home_moneyline'] = line.home_moneyline
                        record['formatted_spread'] = line.formatted_spread
                        record['over_under'] = line.over_under
                        record['provider'] = line.provider
                        # The spreads have different conventions but we want them relative to the away team
                        spread = line.formatted_spread.split(' ')[-1]
                        if spread in ['-null', 'null']:
                            record['spread_line'] = None
                        else:
                            if b.away_team in line.formatted_spread:
                                record['spread_line'] = float(spread)
                            else:
                                record['spread_line'] = -1 * float(spread)
                        if record['away_moneyline'] is None:
                            record['away_moneyline'] = self._impute_money_line_from_spread(record['spread_line'])
                        records.append(record.copy())
                df.append(pd.DataFrame.from_records(records))
        df = pd.concat(df).drop_duplicates().reset_index(drop=True) if df_raw is None else df_raw
        df['gameday'] = pd.to_datetime(df['gameday'])

        # De-dupe from multiple spread providers
        if 'provider' in df.columns:
            df = df.drop('provider', axis=1).drop_duplicates()

        # Arbitrarily take the min spread / away_moneyline
        df['spread_line_min'] = df.groupby('game_id')['spread_line'].transform('min')
        df = df[df['spread_line'] == df['spread_line_min']]
        df = df.drop('spread_line_min', axis=1).drop_duplicates()
        # Take the max over / under for now
        df['over_under'] = df.groupby('game_id')['over_under'].transform('min')
        df['home_moneyline'] = df.groupby('game_id')['home_moneyline'].transform('mean')
        # Impute moneyline from spreads empirical fit to avoid dropping data
        df['away_moneyline'] = df['away_moneyline'].\
            fillna(df['spread_line'].apply(self._impute_money_line_from_spread))
        df['away_moneyline'] = df.groupby('game_id')['away_moneyline'].transform('mean')
        df = df.drop_duplicates().reset_index(drop=True)

        # Drop conferences with proper filter
        college_conferences = ['Big Ten', 'SEC', 'Big 12', 'ACC', 'Pac-12', 'PAC']
        df = df[
            (df['home_conference'].isin(college_conferences))
            &
            (df['away_conference'].isin(college_conferences))
            &
            (~df['spread_line'].isna())
        ]

        # Rename to consistent format
        df = df.rename(columns={'over_under': 'total_line'})

        # Hard de-dupe
        df = df.reset_index(names=['dedupe_col'])
        df = df[df['dedupe_col'] == df.groupby('game_id')['dedupe_col'].transform('min')]

        return df

    def etl(self) -> pd.DataFrame:
        if os.path.exists(os.path.join(self.cache_dir, 'df_training.csv')) and not self.overwrite:
            df = pd.read_csv(os.path.join(self.cache_dir, 'df_training.csv'), parse_dates=['gameday'])
            df = self._add_metrics(df)
            return df
        if self.league == 'nfl':
            # Model training
            logger.info('Downloading Data from Github')
            df = pd.read_csv(self.link_to_data, parse_dates=['gameday'])
            df = df[
                # Regular season only
                (df['game_type'] == 'REG')
                &
                # Not planned
                (~df['away_score'].isna())
            ]
        elif self.league == 'college_football':
            df = self._download_college_football()
        else:
            raise NotImplementedError(self.league)

        # Save to cache
        df.to_csv(os.path.join(self.cache_dir, 'df_training.csv'), index=False)

        # Add metrics off raw data for each game
        df = self._add_metrics(df)
        return df

    @staticmethod
    def _add_metrics(df: pd.DataFrame) -> pd.DataFrame:
        # Metrics
        def _define_spread_favorite(r) -> Optional[float]:
            # away was favorite
            if r['spread_line'] <= 0:
                return r['away_score'] - r['home_score']
            # home was favorite
            elif r['spread_line'] > 0:
                return r['home_score'] - r['away_score']
            else:
                return None
        # Actual spread from perspective of away team
        df['spread_actual'] = df['home_score'] - df['away_score']
        # Difference between actual and odds-spread from perspective of away team
        df['spread_diff'] = (df['home_score'] - df['away_score']) - df['spread_line']
        # odds-spread from perspective of the favorite
        df['spread_favorite'] = df['spread_line'].abs()
        # Actual spread from persepctive of favorite
        df['spread_favorite_actual'] = df.apply(_define_spread_favorite, axis=1)
        # Difference between actual and odds-spread from perspective of favorite team
        df['spread_favorite_diff'] = df['spread_favorite_actual'] - df['spread_favorite']
        # Actual total points
        df['total_actual'] = df['away_score'] + df['home_score']
        # Difference between actual total and odds-total
        df['total_diff'] = df['total_actual'] - df['total_line']
        return df

    @staticmethod
    def label_teams(df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for row in df.to_dict(orient='records'):
            # Away team is favorite
            if row['spread_line'] <= 0:
                record = {re.sub('home', 'underdog', re.sub('away', 'favorite', str(k))): v for k, v in row.items()}
                records.append(record)
            else:
                record = {re.sub('home', 'favorite', re.sub('away', 'underdog', str(k))): v for k, v in row.items()}
                records.append(record)
        return pd.DataFrame.from_records(records)

    def wrangle(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is None:
            df = self.etl()
        # Subset for window past training start
        df__ = df[df['gameday'] > (pd.Timestamp(self.training_start) - pd.Timedelta(days=self.window))]

        records = []
        logger.info(f'Wrangling Data for {self.league}')
        for rdx, row in tqdm(df__.iterrows(), total=df__.shape[0]):

            # Away team of row
            df_ = df[
                (df['away_team'] == row['away_team']) &
                ((row['gameday'] - df['gameday']).dt.days.between(1, self.window))
            ].copy()
            away_wins = (df_['away_score'] > df_['home_score']).sum()
            away_losses = (df_['away_score'] < df_['home_score']).sum()
            away_wins_ats = ((df_['away_score'] + df_['spread_line']) > df_['home_score']).sum()
            away_losses_ats = ((df_['away_score'] + df_['spread_line']) < df_['home_score']).sum()
            away_margin = (df_['away_score'] - df_['home_score']).sum()
            away_margin_ats = (df_['away_score'] + df_['spread_line'] - df_['home_score']).sum()
            away_pf = df_['away_score'].sum()
            away_pa = df_['home_score'].sum()
            away_over = ((df_['away_score'] + df_['home_score']) > df_['total_line']).sum()
            away_under = ((df_['away_score'] + df_['home_score']) < df_['total_line']).sum()
            away_total = away_pf + away_pa
            away_total_ats = (df_['away_score'] + df_['home_score'] - df_['total_line']).sum()
            away_num_games = df_.shape[0]

            df_ = df[
                (df['home_team'] == row['away_team']) &
                ((row['gameday'] - df['gameday']).dt.days.between(1, self.window))
            ].copy()
            home_wins = (df_['home_score'] > df_['away_score']).sum()
            home_losses = (df_['home_score'] < df_['away_score']).sum()
            home_wins_ats = (df_['home_score'] > (df_['away_score'] + df_['spread_line'])).sum()
            home_losses_ats = (df_['home_score'] < (df_['away_score'] + df_['spread_line'])).sum()
            home_margin = (df_['home_score'] - df_['away_score']).sum()
            home_margin_ats = (df_['home_score'] - (df_['away_score'] + df_['spread_line'])).sum()
            home_pf = df_['home_score'].sum()
            home_pa = df_['away_score'].sum()
            home_over = ((df_['away_score'] + df_['home_score']) > df_['total_line']).sum()
            home_under = ((df_['away_score'] + df_['home_score']) < df_['total_line']).sum()
            home_total = home_pf + home_pa
            home_total_ats = (df_['away_score'] + df_['home_score'] - df_['total_line']).sum()
            home_num_games = df_.shape[0]

            record = {
                'game_id': row['game_id'],
                'gameday': row['gameday'],
                # Totals
                'away_team_wins': home_wins + away_wins,
                'away_team_wins_ats': away_wins_ats + home_wins_ats,
                'away_team_losses_ats': away_losses_ats + home_losses_ats,
                'away_team_losses': home_losses + away_losses,
                # Rates
                'away_team_margin': (home_margin + away_margin) / (home_num_games + away_num_games),
                'away_team_margin_ats': (home_margin_ats + away_margin_ats) / (home_num_games + away_num_games),
                'away_team_win_rate': (home_wins + away_wins) / (home_num_games + away_num_games),
                'away_team_win_rate_ats': (home_wins_ats + away_wins_ats) / (home_num_games + away_num_games),
                'away_team_over_rate': (away_over + home_over) / (home_num_games + away_num_games),
                'away_team_under_rate': (away_under + home_under) / (home_num_games + away_num_games),
                'away_team_points_for': (away_pf + home_pf) / (home_num_games + away_num_games),
                'away_team_points_against': (home_pa + away_pa) / (home_num_games + away_num_games),
                'away_team_total_points': (away_total + home_total) / (home_num_games + away_num_games),
                'away_team_total_points_ats': (away_total_ats + home_total_ats) / (home_num_games + away_num_games),
                'away_team_point_differential': (away_pf + home_pf - home_pa - away_pa) / (home_num_games + away_num_games),
                # Lines
                'money_line': self._calc_payout(row['away_moneyline']),
                'away_money_line': self._calc_payout(row['away_moneyline']),
                'away_spread_line': row['spread_line'],  # spread-line is from perspective of away team
            }

            # Home team of row
            df_ = df[
                (df['away_team'] == row['home_team']) &
                ((row['gameday'] - df['gameday']).dt.days.between(1, self.window))
            ].copy()
            away_wins = (df_['away_score'] > df_['home_score']).sum()
            away_losses = (df_['away_score'] < df_['home_score']).sum()
            away_wins_ats = ((df_['away_score'] + df_['spread_line']) > df_['home_score']).sum()
            away_losses_ats = ((df_['away_score'] + df_['spread_line']) < df_['home_score']).sum()
            away_margin = (df_['away_score'] - df_['home_score']).sum()
            away_margin_ats = (df_['away_score'] + df_['spread_line'] - df_['home_score']).sum()
            away_pf = df_['away_score'].sum()
            away_pa = df_['home_score'].sum()
            away_over = ((df_['away_score'] + df_['home_score']) > df_['total_line']).sum()
            away_under = ((df_['away_score'] + df_['home_score']) < df_['total_line']).sum()
            away_num_games = df_.shape[0]
            away_total = away_pf + away_pa
            away_total_ats = (df_['away_score'] + df_['home_score'] - df_['total_line']).sum()

            df_ = df[
                (df['home_team'] == row['home_team']) &
                ((row['gameday'] - df['gameday']).dt.days.between(1, self.window))
            ].copy()
            home_wins = (df_['home_score'] > df_['away_score']).sum()
            home_losses = (df_['home_score'] < df_['away_score']).sum()
            home_wins_ats = (df_['home_score'] > (df_['away_score'] + df_['spread_line'])).sum()
            home_losses_ats = (df_['home_score'] < (df_['away_score'] + df_['spread_line'])).sum()
            home_margin = (df_['home_score'] - df_['away_score']).sum()
            home_margin_ats = (df_['home_score'] - (df_['away_score'] + df_['spread_line'])).sum()
            home_pf = df_['home_score'].sum()
            home_pa = df_['away_score'].sum()
            home_over = ((df_['away_score'] + df_['home_score']) > df_['total_line']).sum()
            home_under = ((df_['away_score'] + df_['home_score']) < df_['total_line']).sum()
            home_num_games = df_.shape[0]
            home_total = home_pf + home_pa
            home_total_ats = (df_['away_score'] + df_['home_score'] - df_['total_line']).sum()

            # Totals
            record['home_team_wins'] = home_wins + away_wins
            record['home_team_losses'] = home_losses + away_losses
            record['home_team_wins_ats'] = home_wins_ats + away_wins_ats
            record['home_team_losses_ats'] = home_losses_ats + away_losses_ats
            # Rates
            record['home_team_margin'] = (home_margin + away_margin) / (home_num_games + away_num_games)
            record['home_team_margin_ats'] = (home_margin_ats + away_margin_ats) / (home_num_games + away_num_games)
            record['home_team_win_rate'] = (home_wins + away_wins) / (home_num_games + away_num_games)
            record['home_team_win_rate_ats'] = (home_wins_ats + away_wins_ats) / (home_num_games + away_num_games)
            record['home_team_over_rate'] = (away_over + home_over) / (home_num_games + away_num_games)
            record['home_team_under_rate'] = (away_under + home_under) / (home_num_games + away_num_games)
            record['home_team_points_for'] = (away_pf + home_pf) / (home_num_games + away_num_games)
            record['home_team_points_against'] = (away_pa + home_pa) / (home_num_games + away_num_games)
            record['home_team_point_differential'] = (home_pf + away_pf - home_pa - away_pa) / (home_num_games + away_num_games)
            record['home_team_total_points'] = (away_total + home_total) / (home_num_games + away_num_games)
            record['home_team_total_points_ats'] = (away_total_ats + home_total_ats) / (home_num_games + away_num_games)
            # Lines
            record['home_money_line'] = self._calc_payout(row['home_moneyline'])
            record['home_spread_line'] = -row['spread_line']  # spread line if from perspective of away team
            records.append(record)

        df_out = pd.DataFrame.from_records(records)
        # Fill na for win-rate
        for col in [
            'away_team_win_rate',
            'away_team_win_rate_ats',
            'away_team_over_rate',
            'away_team_under_rate',
            'home_team_win_rate',
            'home_team_win_rate_ats',
            'home_team_over_rate',
            'home_team_under_rate',
            'home_team_margin',
            'away_team_margin',
            'home_team_margin_ats',
            'away_team_margin_ats',
            'home_team_total_points',
            'away_team_total_points',
            'home_team_total_points_ats',
            'away_team_total_points_ats',
            'home_team'
        ]:
            if col in df_out.columns:
                if df_out[col].isna().mean() > 0:
                    logger.info(f'{col}: NA RATE: {round(df_out[col].isna().mean(), 3)}')
                df_out[col] = df_out[col].fillna(0.)
        df_out = df_out[df_out['gameday'] > self.training_start]

        # Impute these so we can run predictions
        for col in ['spread_actual', 'spread_diff']:
            if col not in df.columns:
                df[col] = None

        # Add back in non-time-dependent features
        df_out = df_out.\
            merge(
                df[[
                    'game_id', 'gameday', 'spread_actual', 'spread_line', 'spread_favorite', 'spread_favorite_actual',
                    'spread_favorite_diff',
                    'spread_diff', 'total_line', 'total_actual', 'total_diff'
                ]],
                on=['game_id', 'gameday']
            )

        # label teams
        df_out = self.label_teams(df_out)

        # Keep one away-specific boolean
        df_out['away_is_favorite'] = (df_out['spread_line'] < 0).astype(int)

        return df_out
