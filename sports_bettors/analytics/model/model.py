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
                    'favorite_team_win_rate_ats',
                    'underdog_team_win_rate_ats',
                    'favorite_money_line',
                    'favorite_spread_line',
                    'total_line',
                    'favorite_team_points_for',
                    'underdog_team_points_for',
                    'favorite_team_points_against',
                    'underdog_team_points_against',
                ]
            },
            'over': {
                'response_col': 'total_actual',
                'line_col': 'total_line',
                'diff_col': 'total_diff',
                'features': [
                    'favorite_team_win_rate_ats',
                    'underdog_team_win_rate_ats',
                    'favorite_money_line',
                    'favorite_spread_line',
                    'total_line',
                    'favorite_team_points_for',
                    'underdog_team_points_for',
                    'favorite_team_points_against',
                    'underdog_team_points_against',
                ]
            }
        },
        'college_football': {
            'spread': {
                'response_col': 'spread_actual',
                'line_col': 'spread_line',
                'diff_col': 'spread_diff',
                'features': [
                    'favorite_team_win_rate_ats',
                    'underdog_team_win_rate_ats',
                    'favorite_money_line',
                    'favorite_spread_line',
                    'favorite_team_points_for',
                    'underdog_team_points_for',
                    'favorite_team_points_against',
                    'underdog_team_points_against',
                ]
            },
            'over': {
                'response_col': 'total_actual',
                'line_col': 'total_line',
                'diff_col': 'total_diff',
                'features': [
                    'favorite_team_win_rate_ats',
                    'underdog_team_win_rate_ats',
                    'favorite_money_line',
                    'favorite_spread_line',
                    'total_line',
                    'favorite_team_points_for',
                    'underdog_team_points_for',
                    'favorite_team_points_against',
                    'underdog_team_points_against',
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
        for col in self.features:
            df = df[~df[col].isna()]
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

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None or self.scaler is None:
            logger.error('No Model and / or scaler')
            raise ValueError()
        return self.model.predict(self.transform(df))

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
