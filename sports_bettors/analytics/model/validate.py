import os
from typing import Optional
import numpy as np
import pandas as pd
import datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

from sports_bettors.analytics.model.model import Model


class Validate(Model):

    @staticmethod
    def _baseline_prob():
        return 0.541

    def validate(self, df_: Optional[pd.DataFrame] = None, df_val: Optional[pd.DataFrame] = None,
                 df: Optional[pd.DataFrame] = None):
        if any([df_ is None, df_val is None, df is None]):
            df_, df_val, df = self.fit_transform()

        df_['preds'] = self.predict_spread(df_)
        df_val['preds'] = self.predict_spread(df_val)
        df['preds'] = self.predict_spread(df)

        # Preds vs Response
        with PdfPages(os.path.join(self.save_dir, 'validate_spreads.pdf')) as pdf:
            bins = np.linspace(-50, 50, 21)
            plt.figure()
            plt.hist(df_val['preds'], alpha=0.5, label='preds', bins=bins)
            plt.hist(df_val[self.response], alpha=0.5, label='Actuals', bins=bins)
            plt.xlabel(self.response)
            plt.title('Preds Distribution')
            plt.grid(True)
            plt.legend()
            pdf.savefig()
            plt.close()

            df_val['res'] = df_val['preds'] - df_val[self.response]
            df_['res'] = df_['preds'] - df_[self.response]
            plt.figure()
            plt.hist(df_val['res'], alpha=0.5, label='test', density=True, bins=bins)
            plt.hist(df_['res'], alpha=0.5, label='train', density=True, bins=bins)
            plt.title('Residuals Distribution')
            plt.grid(True)
            plt.legend()
            pdf.savefig()
            plt.close()

            # Heteroskedasticity
            plt.figure()
            plt.scatter(df_val['preds'], df_val['res'], label='residuals')
            plt.grid(True)
            plt.xlabel('Predictions')
            plt.ylabel('Residuals')
            plt.title('Heteroskedasticity')
            plt.legend()
            pdf.savefig()
            plt.close()

            # Compared to spread
            plt.figure()
            plt.hist(df_val['res'], alpha=0.5, label='residuals', bins=bins)
            plt.hist(df_val['spread_diff'], alpha=0.5, label='spread-diff', bins=bins)
            plt.grid(True)
            plt.legend()
            plt.title('Model vs. Spread')
            plt.xlabel('Error')
            pdf.savefig()
            plt.close()

            # As classifier
            # Logic:
            # If preds (predicted number of points to add to away team to get a tie)
            # is less than spread-line, then we would predict the away team to cover
            # e.g. spread is -7 but we predict -9, or spread is 5 and we predict -3

            # Therefore a "spread against the spread" would be negative for the away team if they are favored to
            # beat the spread
            df['preds_c'] = df['preds'] - df['spread_line']

            # If the actual number of points to add for the away team to get a tie is more than the spread
            # then they lost against the spread
            # e.g. -5 vs -7
            # If the listed spread was greater than observed, then the away team won
            # e.g. -5 vs. -7 or +5 vs +2
            classifier_response = 'away_team_wins_ats'
            df[classifier_response] = (df['spread_line'] > df['spread_actual']).astype(int)
            df_ = df[df['gameday'] < (pd.Timestamp(self.TODAY) - pd.Timedelta(days=self.val_window))].copy()
            df_val = df[df['gameday'] > (pd.Timestamp(self.TODAY) - pd.Timedelta(days=self.val_window))].copy()

            # ROC
            fpr, tpr, thresholds = roc_curve(df_[classifier_response], df_['preds_c'])
            auc = roc_auc_score(df_[classifier_response], df_['preds_c'])
            plt.figure()
            plt.plot(fpr, tpr)
            plt.text(0.2, 0.9, f'AUC: {auc:.3f}\nn={df_val.shape[0]}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title('Train')
            plt.grid(True)
            pdf.savefig()
            plt.close()

            fpr, tpr, thresholds = roc_curve(df_val[classifier_response], df_val['preds_c'])
            auc = roc_auc_score(df_val[classifier_response], df_val['preds_c'])
            plt.figure()
            plt.plot(fpr, tpr)
            plt.text(0.2, 0.9, f'AUC: {auc:.3f}\nn={df_val.shape[0]}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.grid(True)
            plt.title('Test')
            pdf.savefig()
            plt.close()

            precision, recall, thresholds = precision_recall_curve(df_[classifier_response], df_['preds_c'])
            plt.figure()
            plt.plot(thresholds, precision[1:], label='precision')
            plt.plot(thresholds, recall[1:], label='recall')
            plt.hlines(df_val[classifier_response].mean(), min(thresholds), max(thresholds), color='black')
            plt.hlines(0.525, min(thresholds), max(thresholds), color='gray')
            plt.legend()
            plt.grid(True)
            plt.title('Train')
            pdf.savefig()
            plt.close()

            precision, recall, thresholds = precision_recall_curve(df_val[classifier_response], df_val['preds_c'])
            plt.figure()
            plt.plot(thresholds, precision[1:], label='precision')
            plt.plot(thresholds, recall[1:], label='recall')
            plt.legend()
            plt.hlines(df_val[classifier_response].mean(), min(thresholds), max(thresholds), color='black')
            plt.hlines(0.525, min(thresholds), max(thresholds), color='gray')
            plt.grid(True)
            plt.title('Test')
            pdf.savefig()
            plt.close()

            # Win-rate
            records, n_total = [], df_val.shape[0]
            for threshold in np.linspace(-10, 10, 41):
                record = {
                    'threshold': threshold,
                    'fraction_games': df_val[df_val['preds_c'] < threshold].shape[0] / n_total,
                    'win_rate': df_val[(df_val['preds_c']) < threshold][classifier_response].mean(),
                    'team': 'Away'
                }
                records.append(record)
                record = {
                    'threshold': threshold,
                    'fraction_games': df_val[df_val['preds_c'] > threshold].shape[0] / n_total,
                    'win_rate': (1 - df_val[(df_val['preds_c']) > threshold][classifier_response]).mean(),
                    'team': 'Home'
                }
                records.append(record)
            df_plot = pd.DataFrame.from_records(records)
            baseline_prob = self._baseline_prob()
            plt.figure()
            for team, df_ in df_plot.groupby('team'):
                df_ = df_[df_['fraction_games'] > 0.01]
                plt.plot(df_['threshold'], df_['win_rate'], label=team)
            plt.gca().invert_xaxis()
            plt.text(10, 0.92, 'Home Wins Against Spread')
            plt.text(-3, 0.88, 'Away Wins Against Spread')
            plt.legend()
            plt.ylabel('Win Rate')
            plt.xlabel('Predicted Spread on the Vegas-Spread')
            plt.hlines(0.525, -5, 5, color='black')
            plt.hlines(1-0.525, -5, 5, color='black')
            plt.title('Betting Guide')
            plt.grid(True)
            pdf.savefig()
            plt.close()

            plt.figure()
            for team, df_ in df_plot.groupby('team'):
                df_ = df_[df_['win_rate'] > 0.525]
                plt.plot(df_['threshold'], df_['fraction_games'], label=team)
            plt.gca().invert_xaxis()
            plt.text(10, 0.92, 'Home Wins Against Spread')
            plt.text(-3, 0.88, 'Away Wins Against Spread')
            plt.legend()
            plt.ylabel('Fraction of Games with Good Odds')
            plt.xlabel('Predicted Spread on the Vegas-Spread')
            plt.title('Betting Guide: Number of Games')
            plt.grid(True)
            pdf.savefig()
            plt.close()

    def save_results(self):
        import pickle

    def predict_next_week(self) -> pd.DataFrame:
        df = pd.read_csv(self.link_to_data, parse_dates=['gameday'])
        df = self.engineer_features(df)
        df_ = df[
            df['gameday'].between(pd.Timestamp(self.TODAY), pd.Timestamp(self.TODAY) + datetime.timedelta(days=10))
            |
            # Keep this SF game as a test case
            (df['game_id'] == '2023_07_SF_MIN')
        ].copy()
        df_ = df_[(~df_['money_line'].isna() & ~df_['spread_line'].isna()) | (df_['game_id'] == '2023_07_SF_MIN')]

        df_['preds'] = self.predict_spread(df_)
        df_['preds_c'] = df_['spread_line'] - df_['preds']
        print(df_)
        df_.to_csv(os.path.join(os.getcwd(), 'data', 'df_test.csv'), index=False)
        return df_
