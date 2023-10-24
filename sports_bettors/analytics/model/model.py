import datetime
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import shap

from sports_bettors.analytics.model.data import Data
from config import logger


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

    def __init__(self):
        super().__init__()
        self.model = None
        self.scaler = None

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

    def train(self, df: Optional[pd.DataFrame] = None):
        if df is None:
            df, _, _ = self.fit_transform()
        logger.info('Train a Model')
        self.model = SVR(kernel='rbf', C=3, gamma=0.1, epsilon=0.1)
        X, y = pd.DataFrame(self.scaler.transform(df[self.features]), columns=self.features), df[self.response]
        self.model.fit(X, y)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.scaler.transform(df[self.features]), columns=self.features)

    def predict_spread(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None or self.scaler is None:
            logger.error('No Model and / or scaler')
            raise ValueError()
        return self.model.predict(self.transform(df))

    def shap_explain(self, df: pd.DataFrame):
        _, df_, _ = self.fit_transform()
        logger.info('Deriving Explainer')
        explainer = shap.KernelExplainer(self.model.predict, self.transform(df_), nsamples=100, link='identity')
        logger.info('Deriving Shap-Values')
        shap_values = explainer.shap_values(df[self.features].head(2))
        shap.force_plot(explainer.expected_value, shap_values[0, :], df.iloc[0, :], link='logit')