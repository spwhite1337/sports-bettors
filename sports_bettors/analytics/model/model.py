import os
import ast
import time
import pickle
import datetime
from typing import Tuple, Optional, Dict
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedKFold, GridSearchCV

import shap

from sports_bettors.analytics.model.data import Data
from config import logger, Config


class Model(Data):
    val_window = 365
    balance_data = {'nfl': True, 'college_football': False}
    TODAY = datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d')

    n_jobs = 100

    model_data_config = {
        'nfl': {
            'spread': {
                'response_col': 'spread_favorite_actual',
                'line_col': 'spread_favorite',
                'diff_col': 'spread_favorite_diff',
                'features': [
                    'favorite_team_win_rate_ats',
                    'underdog_team_win_rate_ats',
                    # 'money_line',
                    'spread_favorite',
                    'total_line',
                    'favorite_team_points_for',
                    'underdog_team_points_for',
                    'favorite_team_points_against',
                    'underdog_team_points_against',
                    'away_is_favorite',
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
                    'away_is_favorite'
                ]
            }
        },
        'college_football': {
            'spread': {
                'response_col': 'spread_favorite_actual',
                'line_col': 'spread_favorite',
                'diff_col': 'spread_favorite_diff',
                'features': [
                    'favorite_team_win_rate_ats',
                    'underdog_team_win_rate_ats',
                    # 'money_line',
                    'spread_favorite',
                    'favorite_team_points_for',
                    'underdog_team_points_for',
                    'favorite_team_points_against',
                    'underdog_team_points_against',
                    'away_is_favorite',
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
                    'away_is_favorite'
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

    @staticmethod
    def make_resample(_df, column):
        """
        From stack-overflow
        https://stackoverflow.com/questions/53218341/up-sampling-imbalanced-datasets-minor-classes
        """
        dfs_r, dfs_c, bigger, ignore = {}, {}, 0, ''
        for c in _df[column].unique():
            dfs_c[c] = _df[_df[column] == c]
            if dfs_c[c].shape[0] > bigger:
                bigger = dfs_c[c].shape[0]
                ignore = c
        for c in dfs_c:
            if c == ignore:
                continue
            dfs_r[c] = resample(dfs_c[c], replace=True, n_samples=bigger - dfs_c[c].shape[0], random_state=0)
        return pd.concat([dfs_r[c] for c in dfs_r] + [_df])

    def fit_transform(self, df: Optional[pd.DataFrame] = None, val: bool = False
                      ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if df is None:
            df = self.wrangle()

        # Drop nas
        df = df[~df[self.line_col].isna() & ~df[self.response_col].isna()]
        for col in self.features:
            df = df[~df[col].isna()]

        # Train test split
        df_ = df[df['gameday'] < (pd.Timestamp(self.TODAY) - pd.Timedelta(days=self.val_window))].copy()
        df_val = df[df['gameday'] > (pd.Timestamp(self.TODAY) - pd.Timedelta(days=self.val_window))].copy()

        # Balance Data
        if self.balance_data[self.league] and not val:
            df_['balance'] = df_[self.response_col] > df_[self.line_col]
            df_val['balance'] = df_val[self.response_col] > df_val[self.line_col]
            df_ = self.make_resample(df_, 'balance')
            df_val = self.make_resample(df_val, 'balance')
            df = pd.concat([df_, df_val]).reset_index(drop=True).drop('balance', axis=1)

        # Scale features
        self.scaler = StandardScaler()
        self.scaler.fit(df_[self.features])

        return df_, df_val, df

    def get_hyper_params(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
        # Define model
        model = Pipeline([('model', SVR(kernel='rbf', gamma=0.1, epsilon=0.1))])
        parameters = {
            'model__gamma': [0.05, 0.1, 0.2],
            'model__epsilon': [0.05, 0.1, 0.2],
            'model__C': [0.1, 0.5, 1, 2, 3, 5, 10]
        }
        grid = GridSearchCV(
            model,
            cv=3,
            param_grid=parameters,
            scoring='r2',
            return_train_score=True,
            verbose=1,
            n_jobs=None
        )
        logger.info(f'Running Grid Search for {self.league} on {self.response}')
        grid.fit(X, y)
        df = pd.DataFrame().from_dict(grid.cv_results_)
        df = df[df['mean_test_score'] == df['mean_test_score'].max()]
        return df['params'].iloc[0]

    def train(self, df: Optional[pd.DataFrame] = None, df_val: Optional[pd.DataFrame] = None):
        if df is None or df_val is None:
            df, df_val, _ = self.fit_transform()

        # Train / test split
        X, y = pd.DataFrame(self.scaler.transform(df[self.features]), columns=self.features), df[self.response_col]

        # Get hyper-params
        hyper_params = self.get_hyper_params(X, y)
        logger.info(f'Training a Model for {self.league} on {self.response}')
        self.model = Pipeline([
            ('model', SVR(
                kernel='rbf',
                C=hyper_params['model__C'],
                gamma=hyper_params['model__gamma'],
                epsilon=hyper_params['model__epsilon']
            ))
        ])
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
