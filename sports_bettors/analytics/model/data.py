from typing import Optional
import pandas as pd
from tqdm import tqdm
import datetime

from sports_bettors.analytics.eda.eda import Eda
from config import logger


class Data(Eda):
    training_years = 5
    link_to_data = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
    window = 365

    def __init__(self):
        super().__init__()
        self.training_start = datetime.datetime.strftime(
            datetime.datetime.today() - datetime.timedelta(days=self.training_years * 365),
            '%Y-%m-%d',
        )

    def etl(self) -> pd.DataFrame:
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
