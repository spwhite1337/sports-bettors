import datetime
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import shap
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, precision_recall_curve

from sports_bettors.analytics.model.data import Data
from config import logger


class Model(Data):
    val_window = 365
    TODAY = datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d')

    features = [
        'away_team_wins_ats_past_month',
        'away_team_losses_ats_past_month',
        'home_team_wins_ats_past_month',
        'home_team_losses_ats_past_month',
        'money_line',
        'spread_line',
        'away_team_points_for_past_month',
        'home_team_points_for_past_month',
        'away_team_points_against_past_month',
        'home_team_points_against_past_month',
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
        X, y = df_[self.features].values, df_[self.response].values
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        return df_, df_val, df

    def train(self, df: pd.DataFrame):
        self.model = SVR(kernel='rbf', C=3, gamma=0.1, epsilon=0.1)
        X, y = self.scaler.transform(df[self.features]), df[self.response]
        self.model.fit(X, y)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.scaler.transform(df[self.features])

    def predict_spread(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None or self.scaler is None:
            logger.error('No Model and / or scaler')
            raise ValueError()

        return self.model.predict(self.transform(df))

    def shap_explain(self, df: pd.DataFrame):
        df_, _, _ = self.fit_transform()
        logger.info('Deriving Explainer')
        explainer = shap.KernelExplainer(self.model.predict, self.transform(df_), nsamples=100, link='identity')
        logger.info('Deriving Shap-Values')
        shap_values = explainer.shap_values(df[self.features].head(2))
        print(type(shap_values))
        shap.force_plot(explainer.expected_value, shap_values[0, :], df.iloc[0, :], link='logit')