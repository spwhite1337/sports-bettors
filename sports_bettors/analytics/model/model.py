import datetime
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, precision_recall_curve

from sports_bettors.analytics.model.data import Data


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

        df_ = df[df['gameday'] < (pd.Timestamp(self.TODAY) - pd.Timedelta(days=self.val_window))].copy()
        df_val = df[df['gameday'] > (pd.Timestamp(self.TODAY) - pd.Timedelta(days=self.val_window))].copy()

        X, y = df_[self.features].values, df_[self.response].values
        X_val = df_val[self.features].values
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        X_val = self.scaler.transform(X_val)
        X_all = self.scaler.transform(df[self.features].values)
        return X, X_val, X_all

    def train(self, X: pd.DataFrame, y: np.ndarray):
        self.model = SVR(kernel='rbf', C=3, gamma=0.1, epsilon=0.1)
        self.model.fit(X, y)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.scaler.transform(df[self.features])

    def predict_spread(self, df: pd.DataFrame) -> pd.Series:
        return self.model.predict(self.transform(df))
