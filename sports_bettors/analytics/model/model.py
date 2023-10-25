import os
import time
import pickle
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

    model_data_config = {
        'nfl': {
            'spread': {
                'response_col': 'spread_actual',
                'line_col': 'spread_line',
                'diff_col': 'spread_diff',
                'features': [
                    'away_team_win_rate_ats',
                    'home_team_win_rate_ats',
                    'money_line',
                    'spread_line',
                    'total_line',
                    'away_team_points_for',
                    'home_team_points_for',
                    'away_team_points_against',
                    'home_team_points_against',
                ]
            },
            'over': {
                'response_col': 'total_actual',
                'line_col': 'total_line',
                'diff_col': 'total_diff',
                'features': [
                    'away_team_win_rate_ats',
                    'home_team_win_rate_ats',
                    'money_line',
                    'spread_line',
                    'total_line',
                    'away_team_points_for',
                    'home_team_points_for',
                    'away_team_points_against',
                    'home_team_points_against',
                ]
            }
        },
        'college_football': {
            'spread': {
                'response_col': 'spread_actual',
                'line_col': 'spread_line',
                'diff_col': 'spread_diff',
                'features': [
                    'away_team_win_rate_ats',
                    'home_team_win_rate_ats',
                    'money_line',
                    'spread_line',
                    'away_team_points_for',
                    'home_team_points_for',
                    'away_team_points_against',
                    'home_team_points_against',
                ]
            },
            'over': {
                'response_col': 'total_actual',
                'line_col': 'total_line',
                'diff_col': 'total_diff',
                'features': [
                    'away_team_win_rate_ats',
                    'home_team_win_rate_ats',
                    'money_line',
                    'spread_line',
                    'away_team_points_for',
                    'home_team_points_for',
                    'away_team_points_against',
                    'home_team_points_against',
                ]
            }
        }
    }

    def __init__(self, league: str = 'nfl', response: str = 'spread', overwrite: bool = False):
        super().__init__(league=league, overwrite=overwrite)
        self.model = None
        self.scaler = None
        self.response = response
        self.features = self.model_data_config[self.league][self.response]['features']
        self.response_col = self.model_data_config[self.league][self.response]['response_col']
        self.line_col = self.model_data_config[self.league][self.response]['line_col']
        self.diff_col = self.model_data_config[self.league][self.response]['diff_col']
        self.save_dir = os.path.join(os.getcwd(), 'docs', 'model', self.league, self.response)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.model_dir = os.path.join(os.getcwd(), 'data', 'sports_bettors', 'models', self.league, self.response)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def fit_transform(self, df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if df is None:
            df = self.engineer_features()
        # Drop nas
        df = df[~df[self.line_col].isna() & ~df[self.response_col].isna()]
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
        X, y = pd.DataFrame(self.scaler.transform(df[self.features]), columns=self.features), df[self.response_col]
        self.model.fit(X, y)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.scaler.transform(df[self.features]), columns=self.features)

    def predict_spread(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None or self.scaler is None:
            logger.error('No Model and / or scaler')
            raise ValueError()
        return self.model.predict(self.transform(df))

    def predict_next_week(self) -> Tuple[pd.DataFrame, str]:
        if self.league == 'nfl':
            df = pd.read_csv(self.link_to_data, parse_dates=['gameday'])
            df = df[df['gameday'] > (pd.Timestamp(self.TODAY) - datetime.timedelta(days=self.window))]
        elif self.league == 'college_football':
            df = self._download_college_football(predict=True)
        else:
            raise NotImplementedError(self.league)

        # Engineer features from raw
        df = self.calcs(df)
        df = self.engineer_features(df)

        # Filter for predictions
        df = df[
            # Next week of League
            df['gameday'].between(pd.Timestamp(self.TODAY), pd.Timestamp(self.TODAY) + datetime.timedelta(days=10))
            |
            # Keep this SF game as a test case
            (df['game_id'] == '2023_07_SF_MIN')
            |
            # Keep a college game as a test case
            (df['game_id'] == 'COLLEGE_TEST_GAME')
        ].copy()

        # Filter for bad features
        for feature in self.features:
            test_games = ['2023_07_SF_MIN']
            df = df[~df[feature].isna() | (df['game_id'].isin(test_games))]

        # Margin of victory for home-team is like a spread for away team
        df['predicted_margin_of_victory_for_home_team'] = self.predict_spread(df)
        df['spread_against_spread'] = df['predicted_margin_of_victory_for_home_team'] - df['spread_line']

        # Label bets
        df['Bet_ATS'] = df.apply(lambda r: Config.label_bet_ats(self.league, r['spread_against_spread']), axis=1)

        # Print results
        print(df[[
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
        fn = f'df_{int(time.time())}.csv'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df.to_csv(os.path.join(save_dir, fn), index=False)

        return df, fn

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

    def shap_explain(self, df: pd.DataFrame):
        # Example plot for jupyter analysis
        _, df_, _ = self.fit_transform()
        logger.info('Deriving Explainer')
        explainer = shap.KernelExplainer(self.model.predict, self.transform(df_), nsamples=100, link='identity')
        logger.info('Deriving Shap-Values')
        shap_values = explainer.shap_values(df[self.features].head(2))
        shap.force_plot(explainer.expected_value, shap_values[0, :], df.iloc[0, :], link='logit')
