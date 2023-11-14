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
from sklearn.model_selection import GroupKFold, GridSearchCV

import shap

from sports_bettors.analytics.model.data import Data
from config import logger, Config


class Model(Data):
    val_window = 365
    balance_data = {'nfl': True, 'college_football': True}
    TODAY = datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d')

    n_jobs = 100

    model_data_config = {
        'nfl': {
            'spread': {
                'response_col': 'spread_favorite_actual',
                'line_col': 'spread_favorite',
                'diff_col': 'spread_favorite_diff',
                'features': [
                    # Lines
                    'spread_favorite',
                    'total_line',
                    # 'away_is_favorite',
                    # Win-rate ATS / Over
                    'favorite_team_win_rate_ats',
                    'underdog_team_win_rate_ats',
                    'favorite_margin_ats',
                    'underdog_margin_ats',

                    # PF and PA
                    'favorite_team_over_rate',
                    'underdog_team_over_rate',
                    'favorite_team_total_points_ats',
                    'underdog_team_total_points_ats',
                    # 'favorite_team_points_for',
                    # 'underdog_team_points_for',
                    # 'favorite_team_points_against',
                    # 'underdog_team_points_against',
                ]
            },
            'over': {
                'response_col': 'total_actual',
                'line_col': 'total_line',
                'diff_col': 'total_diff',
                'features': [
                    # Lines
                    'spread_favorite',
                    'total_line',
                    # 'away_is_favorite',
                    # Win-rate ATS
                    'favorite_team_win_rate_ats',
                    'underdog_team_win_rate_ats',
                    'favorite_margin_ats',
                    'underdog_margin_ats',

                    # PF and PA
                    'favorite_team_over_rate',
                    'underdog_team_over_rate',
                    'favorite_team_total_points_ats',
                    'underdog_team_total_points_ats',
                    # 'favorite_team_points_for',
                    # 'underdog_team_points_for',
                    # 'favorite_team_points_against',
                    # 'underdog_team_points_against',
                ]
            }
        },
        'college_football': {
            'spread': {
                'response_col': 'spread_favorite_actual',
                'line_col': 'spread_favorite',
                'diff_col': 'spread_favorite_diff',
                'features': [
                    # Lines
                    'spread_favorite',
                    'total_line',
                    # 'away_is_favorite',
                    # Win-rate ATS
                    'favorite_team_win_rate_ats',
                    'underdog_team_win_rate_ats',
                    'favorite_margin_ats',
                    'underdog_margin_ats',

                    # PF and PA
                    'favorite_team_over_rate',
                    'underdog_team_over_rate',
                    'favorite_team_total_points_ats',
                    'underdog_team_total_points_ats',
                    # 'favorite_team_points_for',
                    # 'underdog_team_points_for',
                    # 'favorite_team_points_against',
                    # 'underdog_team_points_against',
                ]
        },
            'over': {
                'response_col': 'total_actual',
                'line_col': 'total_line',
                'diff_col': 'total_diff',
                'features': [
                    # Lines
                    'spread_favorite',
                    'total_line',
                    # 'away_is_favorite',
                    # Win-rate ATS
                    'favorite_team_win_rate_ats',
                    'underdog_team_win_rate_ats',
                    'favorite_margin_ats',
                    'underdog_margin_ats',
                    # PF and PA
                    'favorite_team_over_rate',
                    'underdog_team_over_rate',
                    'favorite_team_total_points_ats',
                    'underdog_team_total_points_ats',
                    # 'favorite_team_points_for',
                    # 'underdog_team_points_for',
                    # 'favorite_team_points_against',
                    # 'underdog_team_points_against',
                ]
            }
        }
    }

    def __init__(self, league: str = 'nfl', response: str = 'spread', overwrite: bool = False):
        super().__init__(league=league, overwrite=overwrite)
        self.model = None
        self.scaler = None
        self.hyper_params = {}
        self.opt_metric = None
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
            if bigger == dfs_c[c].shape[0]:
                dfs_r[c] = dfs_c[c]
            else:
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

        if self.response == 'spread':
            # Clip training data so it doesn't over-penalize bad spreads
            df_[self.response_col] = df_[self.response_col].clip(-7, 7)
        elif self.response == 'over':
            # Clip training data so it doesn't over-penalize bad over totals
            df_[self.response_col] = df_[self.response_col].clip(df_[self.line_col] - 20, df_[self.line_col] + 20)

        return df_, df_val, df

    def get_hyper_params(self, X: pd.DataFrame, y: pd.DataFrame, group: pd.Series) -> Dict[str, float]:
        # if self.response == 'spread' and self.league == 'nfl':
        #     self.opt_metric = -9999
        #     return {
        #         'model__kernel': 'rbf',
        #         'model__gamma': 0.2,
        #         'model__epsilon': 0.1,
        #         'model__C': 1
        #     }

        # Define model
        model = Pipeline([('model', SVR())])
        parameters = {
            'model__kernel': ['rbf', 'poly', 'sigmoid'],
            'model__gamma': ['scale', 'auto'],
            'model__epsilon': [0.05, 0.1, 0.2],
            'model__C': [0.1, 0.5, 1, 2, 3, 5, 10]
        }
        gkf = GroupKFold(n_splits=group.nunique())
        grid = GridSearchCV(
            model,
            cv=gkf.split(X, y, group),
            param_grid=parameters,
            # scoring='r2',
            scoring='neg_mean_squared_error',
            return_train_score=True,
            verbose=1,
            n_jobs=None
        )
        logger.info(f'Running Grid Search for {self.league} on {self.response}')
        grid.fit(X, y)
        df = pd.DataFrame().from_dict(grid.cv_results_)
        self.opt_metric = df['mean_test_score'].max()
        df = df[df['mean_test_score'] == self.opt_metric]
        return df['params'].iloc[0]

    def train(self, df: Optional[pd.DataFrame] = None, df_val: Optional[pd.DataFrame] = None):
        if df is None or df_val is None:
            df, df_val, _ = self.fit_transform()

        # Train / test split
        X, y = pd.DataFrame(self.scaler.transform(df[self.features]), columns=self.features), df[self.response_col]

        # Get hyper-params
        df['group_col'] = df['gameday'].dt.year
        self.hyper_params = self.get_hyper_params(X, y, df['group_col'])
        logger.info(f'Training a Model for {self.league} on {self.response}')
        self.model = Pipeline([
            ('model', SVR(
                kernel=self.hyper_params['model__kernel'],
                C=self.hyper_params['model__C'],
                gamma=self.hyper_params['model__gamma'],
                epsilon=self.hyper_params['model__epsilon'],
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
