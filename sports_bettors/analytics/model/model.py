import os
import time
import datetime
from typing import Tuple, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import shap

from sports_bettors.analytics.model.data import Data
from config import logger, Config


class Model(Data):
    val_window = 365
    TODAY = datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d')

    features = [
        # 'away_team_wins_ats',
        # 'away_team_losses_ats',
        # 'home_team_wins_ats',
        # 'home_team_losses_ats',
        'away_team_win_rate_ats',
        'home_team_win_rate_ats',
        'money_line',
        'spread_line',
        'away_team_points_for',
        'home_team_points_for',
        'away_team_points_against',
        'home_team_points_against',
        # 'away_team_point_differential',
        # 'home_team_point_differential'
    ]
    response = 'spread_actual'

    def __init__(self, league: str = 'nfl', overwrite: bool = False):
        super().__init__(league=league, overwrite=overwrite)
        self.model = None
        self.scaler = None
        self.save_dir = os.path.join(os.getcwd(), 'docs', 'model', self.league)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def fit_transform(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if df is None:
            df = self.engineer_features()
        # Train test split
        df_ = df[df['gameday'] < (pd.Timestamp(self.TODAY) - pd.Timedelta(days=self.val_window))].copy()
        df_val = df[df['gameday'] > (pd.Timestamp(self.TODAY) - pd.Timedelta(days=self.val_window))].copy()
        # Scale features
        self.scaler = StandardScaler()
        self.scaler.fit(df_[self.features])
        return df_, df_val, df

    def get_hyper_params(self) -> Dict[str, float]:
        if self.league == 'nfl':
            return {'C': 3}
        elif self.league == 'college_football':
            return {'C': 3}

    def train(self, df: Optional[pd.DataFrame] = None):
        if df is None:
            df, _, _ = self.fit_transform()
        logger.info('Train a Model')
        self.model = SVR(
            kernel='rbf',
            gamma=0.1,
            epsilon=0.1,
            **self.get_hyper_params()
        )
        X, y = pd.DataFrame(self.scaler.transform(df[self.features]), columns=self.features), df[self.response]
        self.model.fit(X, y)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.scaler.transform(df[self.features]), columns=self.features)

    def predict_spread(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None or self.scaler is None:
            logger.error('No Model and / or scaler')
            raise ValueError()
        return self.model.predict(self.transform(df))

    def predict_next_week(self) -> pd.DataFrame:
        if self.league == 'nfl':
            df = pd.read_csv(self.link_to_data, parse_dates=['gameday'])
            df = df[df['gameday'] > (pd.Timestamp(self.TODAY) - datetime.timedelta(days=self.window))]
        elif self.league == 'college_football':
            df = self._download_college_football(predict=True)
        else:
            raise NotImplementedError(self.league)

        df = self.engineer_features(df)
        df_ = df[
            # Next week of League
            df['gameday'].between(pd.Timestamp(self.TODAY), pd.Timestamp(self.TODAY) + datetime.timedelta(days=10))
            |
            # Keep this SF game as a test case
            (df['game_id'] == '2023_07_SF_MIN')
            |
            # Keep a college game as a test case
            (df['game_id'] == 'COLLEGE_TEST_GAME')
            ].copy()
        df_ = df_[(~df_['money_line'].isna() & ~df_['spread_line'].isna()) | (df_['game_id'] == '2023_07_SF_MIN')]

        # Margin of victory for home-team is like a spread for away team
        df_['predicted_margin_of_victory_for_home_team'] = self.predict_spread(df_)
        df_['spread_against_spread'] = df_['predicted_margin_of_victory_for_home_team'] - df_['spread_line']

        # Label bets
        df_['Bet_ATS'] = df_.apply(lambda r: Config.label_bet_ats(self.league, r['spread_against_spread']), axis=1)

        # Print results
        print(df_[[
            'game_id',
            'gameday',
            'spread_actual',
            'spread_line',
            'money_line',
            'predicted_margin_of_victory_for_home_team',
            'spread_against_spread',
            'Bet_ATS'
        ]])

        # Save results
        save_dir = os.path.join(os.getcwd(), 'data', 'predictions', self.league)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df_.to_csv(os.path.join(save_dir, f'df_{int(time.time())}.csv'), index=False)

        return df_

    def shap_explain(self, df: pd.DataFrame):
        # Example plot for jupyter analysis
        _, df_, _ = self.fit_transform()
        logger.info('Deriving Explainer')
        explainer = shap.KernelExplainer(self.model.predict, self.transform(df_), nsamples=100, link='identity')
        logger.info('Deriving Shap-Values')
        shap_values = explainer.shap_values(df[self.features].head(2))
        shap.force_plot(explainer.expected_value, shap_values[0, :], df.iloc[0, :], link='logit')