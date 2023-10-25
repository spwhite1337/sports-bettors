import re
import os
import pickle
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
    link_to_data = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
    window = 365

    college_conferences = ['ACC', 'B12', 'B1G', 'SEC', 'Pac-10', 'PAC', 'Ind']

    def __init__(self, league: str = 'nfl', overwrite: bool = False):
        super().__init__()
        self.training_start = datetime.datetime.strftime(
            datetime.datetime.today() - datetime.timedelta(days=self.training_years * 365),
            '%Y-%m-%d',
        )
        self.overwrite = overwrite
        assert league in ['nfl', 'college_football']
        self.league = league
        self.model_dir = os.path.join(os.getcwd(), 'data', 'sports_bettors', 'models', self.league)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
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

    def _download_college_football(self, predict: bool = False) -> pd.DataFrame:
        # Pull data from https://github.com/CFBD/cfbd-python
        # As of 10/2023 it is "free to use without restrictions"

        configuration = cfbd.Configuration()
        configuration.api_key['Authorization'] = os.environ['API_KEY_COLLEGE_API']
        configuration.api_key_prefix['Authorization'] = 'Bearer'
        api_instance = cfbd.BettingApi(cfbd.ApiClient(configuration))
        current_year = datetime.datetime.today().year
        if not predict:
            years = list(np.linspace(current_year - self.training_years - 1, current_year, self.training_years + 2))
        else:
            years = list(np.linspace(current_year - 1, current_year, 2))
        season_type = 'regular'
        df = []
        for year in tqdm(years):
            for conference in tqdm(self.college_conferences):
                # Rest a bit for the API because it is free
                time.sleep(2)
                try:
                    api_response = api_instance.get_lines(year=year, season_type=season_type, conference=conference)
                except:
                    if predict:
                        return pd.read_csv(os.path.join(self.cache_dir, 'df_training.csv'), parse_dates=['gameday'])
                    else:
                        raise Exception
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
        df = pd.concat(df).drop_duplicates().reset_index(drop=True)
        df['gameday'] = pd.to_datetime(df['gameday']).dt.date

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
        # Impute moneyline from spreads empircal fit to avoid dropping data
        df['away_moneyline'] = df['away_moneyline'].\
            fillna(df['spread_line'].apply(self._impute_money_line_from_spread))
        df['away_moneyline'] = df.groupby('game_id')['away_moneyline'].transform('mean')
        df = df.drop_duplicates().reset_index(drop=True)

        # Drop conferences with proper filter
        college_conferences = ['Big Ten', 'SEC', 'Big 12', 'ACC', 'Pac-12', 'PAC', 'FBS Independents']
        df = df[
            (df['home_conference'].isin(college_conferences))
            &
            (df['away_conference'].isin(college_conferences))
            &
            (~df['spread_line'].isna())
        ]
        return df

    def etl(self) -> pd.DataFrame:
        if os.path.exists(os.path.join(self.cache_dir, 'df_training.csv')) and not self.overwrite:
            return pd.read_csv(os.path.join(self.cache_dir, 'df_training.csv'), parse_dates=['gameday'])
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
        return df

    def calcs(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is None:
            df = self.etl()
        # Metrics
        df['spread_actual'] = df['home_score'] - df['away_score']
        df['spread_diff'] = df['away_score'] + df['spread_line'] - df['home_score']
        df['total_actual'] = df['away_score'] + df['home_score']
        df['off_spread'] = (df['spread_actual'] - df['spread_line'])
        df['off_total'] = (df['total_actual'] - df['total_line'])
        return df

    def engineer_features(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is None:
            df = self.calcs()

        records = []
        # Subset for window past training start
        df__ = df[df['gameday'] > (pd.Timestamp(self.training_start) - pd.Timedelta(days=self.window))]
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
            away_pf = df_['away_score'].sum()
            away_pa = df_['home_score'].sum()
            away_total = away_pf + away_pa

            df_ = df[
                (df['home_team'] == row['away_team']) &
                ((row['gameday'] - df['gameday']).dt.days.between(1, self.window))
            ].copy()
            home_wins = (df_['home_score'] > df_['away_score']).sum()
            home_losses = (df_['home_score'] < df_['away_score']).sum()
            home_wins_ats = (df_['home_score'] > (df_['away_score'] + df_['spread_line'])).sum()
            home_losses_ats = (df_['home_score'] < (df_['away_score'] + df_['spread_line'])).sum()
            home_pf = df_['home_score'].sum()
            home_pa = df_['away_score'].sum()
            home_total = home_pf + home_pa

            record = {
                'game_id': row['game_id'],
                'gameday': row['gameday'],
                'away_team_wins': home_wins + away_wins,
                'away_team_wins_ats': away_wins_ats + home_wins_ats,
                'away_team_losses_ats': away_losses_ats + home_losses_ats,
                'away_team_losses': home_losses + away_losses,
                'away_team_win_rate': (home_wins + away_wins) / (home_wins + away_wins + home_losses + away_losses),
                'away_team_win_rate_ats': (home_wins_ats + away_wins_ats) / (home_wins_ats + away_wins_ats + home_losses_ats + away_losses_ats),
                'away_team_points_for': away_pf + home_pf,
                'away_team_points_against': home_pa + away_pa,
                'away_team_total_points': away_total + home_total,
                'away_team_point_differential': away_pf + home_pf - home_pa - away_pa,
                'money_line': self._calc_payout(row['away_moneyline'])
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
            away_pf = df_['away_score'].sum()
            away_pa = df_['home_score'].sum()
            away_total = away_pf + away_pa

            df_ = df[
                (df['home_team'] == row['home_team']) &
                ((row['gameday'] - df['gameday']).dt.days.between(1, self.window))
            ].copy()
            home_wins = (df_['home_score'] > df_['away_score']).sum()
            home_losses = (df_['home_score'] < df_['away_score']).sum()
            home_wins_ats = (df_['home_score'] > (df_['away_score'] + df_['spread_line'])).sum()
            home_losses_ats = (df_['home_score'] < (df_['away_score'] + df_['spread_line'])).sum()
            home_pf = df_['home_score'].sum()
            home_pa = df_['away_score'].sum()
            home_total = home_pf + home_pa

            record['home_team_wins'] = home_wins + away_wins
            record['home_team_losses'] = home_losses + away_losses
            record['home_team_wins_ats'] = home_wins_ats + away_wins_ats
            record['home_team_losses_ats'] = home_losses_ats + away_losses_ats
            record['home_team_win_rate'] = (home_wins + away_wins) / (home_wins + away_wins + home_losses + away_losses)
            record['home_team_win_rate_ats'] = (home_wins_ats + away_wins_ats) / (home_wins_ats + away_wins_ats + home_losses_ats + away_losses_ats)
            record['home_team_points_for'] = away_pf + home_pf
            record['home_team_points_against'] = home_pa + home_pa
            record['home_team_point_differential'] = home_pf + away_pf - home_pa - away_pa
            record['home_team_total_points'] = away_total + home_total
            records.append(record)

        df_out = pd.DataFrame.from_records(records)
        # Fill na for win-rate
        for col in [
            'away_team_win_rate',
            'away_team_win_rate_ats',
            'home_team_win_rate',
            'home_team_win_rate_ats',
        ]:
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(0.)
        df_out = df_out[df_out['gameday'] > self.training_start]
        for col in ['spread_actual', 'spread_diff']:
            if col not in df.columns:
                df[col] = None
        df_out = df_out.\
            merge(
                df[['game_id', 'gameday', 'spread_actual', 'spread_line', 'spread_diff']],
                on=['game_id', 'gameday']
            )
        return df_out

    def save_results(self):
        filepath = os.path.join(self.model_dir, 'model.pkl')
        with open(filepath, 'wb') as fp:
            pickle.dump(self, fp)

    def load_results(self, model_dir: Optional[str] = None):
        model_dir = self.model_dir if model_dir is None else model_dir
        filepath = os.path.join(model_dir, 'model.pkl')
        if not os.path.exists(filepath):
            print('No Model')
            return None
        with open(filepath, 'rb') as fp:
            obj = pickle.load(fp)
        return obj
