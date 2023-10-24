import os
import pickle
from typing import Optional
import numpy as np
import pandas as pd
import datetime
import shap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

from sports_bettors.analytics.model.model import Model
from config import logger


class Validate(Model):

    def __init__(self):
        super().__init__()
        self.model_dir = os.path.join(os.getcwd(), 'data', 'sports_bettors', 'models')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    @staticmethod
    def _baseline_prob():
        return 0.541

    def validate(self, df_: Optional[pd.DataFrame] = None, df_val: Optional[pd.DataFrame] = None,
                 df: Optional[pd.DataFrame] = None, run_shap: bool = False):
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

            # Shap-values
            if run_shap:
                logger.info('Shap')
                df_plot = df_val[self.features].sample(100) if df_val.shape[0] > 100 else df_val[self.features]
                explainer = shap.KernelExplainer(self.model.predict, self.transform(df_val), nsamples=100, link='identity')
                shap_values = explainer.shap_values(self.transform(df_plot))
                plt.figure()
                shap.summary_plot(shap_values, features=self.features, plot_type='bar', show=False)
                plt.tight_layout()
                pdf.savefig()
                plt.close()
                # Shap by features
                for feature in self.features:
                    plt.figure()
                    shap.dependence_plot(
                        ind=feature,
                        shap_values=shap_values[1] if isinstance(shap_values, list) else shap_values,
                        features=self.transform(df_plot[self.features]),
                    )
                    plt.tight_layout()
                    plt.title(feature)
                    pdf.savefig()
                    plt.close()
            # As classifier
            # Logic:
            # If preds (predicted number of points to add to away team to get a tie)
            # is less than spread-line, then we would predict the away team to cover
            # e.g. spread is -7 but we predict -9, or spread is 5 and we predict -3
            # If it is more than the spread-line, then we would predict the home team to cover
            # e.g. spread is -5 but we predict +1, or spread is +7 and we predict +10
            # In this case, the away team is favored but we see them as underdogs relative to the spread

            # Therefore a "spread against the spread" is the away team's expected margin of victory against the
            # spread.
            # Our advantage is that we just need to get a moneyline bet off this new spread-spread
            # preds_c is the amount you need to add to the spread to get it "right"
            df['preds_c'] = df['preds'] - df['spread_line']

            # The away team wins against the spread if the spread line was larger than the margin of victory
            # E.g. spread is -7, but actual spread was -12
            # e.g. spread is +3 but actual spread was +1
            classifier_response = 'away_team_wins_ats'
            df[classifier_response] = (df['spread_line'] > df['spread_actual']).astype(int)
            df_ = df[df['gameday'] < (pd.Timestamp(self.TODAY) - pd.Timedelta(days=self.val_window))].copy()
            df_val = df[df['gameday'] > (pd.Timestamp(self.TODAY) - pd.Timedelta(days=self.val_window))].copy()

            # ROC
            # 1- response so polarity is "normal"
            fpr, tpr, thresholds = roc_curve(1-df_[classifier_response], df_['preds_c'])
            auc = roc_auc_score(1-df_[classifier_response], df_['preds_c'])
            plt.figure()
            plt.plot(fpr, tpr)
            plt.text(0.2, 0.9, f'AUC: {auc:.3f}\nn={df_val.shape[0]}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title('Train')
            plt.grid(True)
            pdf.savefig()
            plt.close()

            fpr, tpr, thresholds = roc_curve(1-df_val[classifier_response], df_val['preds_c'])
            auc = roc_auc_score(1-df_val[classifier_response], df_val['preds_c'])
            plt.figure()
            plt.plot(fpr, tpr)
            plt.text(0.2, 0.9, f'AUC: {auc:.3f}\nn={df_val.shape[0]}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.grid(True)
            plt.title('Test')
            pdf.savefig()
            plt.close()

            precision, recall, thresholds = precision_recall_curve(1-df_[classifier_response], df_['preds_c'])
            plt.figure()
            plt.plot(thresholds, precision[1:], label='precision')
            plt.plot(thresholds, recall[1:], label='recall')
            plt.hlines(1-df_val[classifier_response].mean(), min(thresholds), max(thresholds), color='black')
            plt.hlines(0.525, min(thresholds), max(thresholds), color='gray')
            plt.legend()
            plt.grid(True)
            plt.title('Train')
            pdf.savefig()
            plt.close()

            precision, recall, thresholds = precision_recall_curve(1-df_val[classifier_response], df_val['preds_c'])
            plt.figure()
            plt.plot(thresholds, precision[1:], label='precision')
            plt.plot(thresholds, recall[1:], label='recall')
            plt.legend()
            plt.hlines(1-df_val[classifier_response].mean(), min(thresholds), max(thresholds), color='black')
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
                    # preds_c is "the amount we need to add to the spread-line to get it "right"
                    # If preds_c is negative, then we are expecting the away team to do better
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
        filepath = os.path.join(self.model_dir, 'model.pkl')
        with open(filepath, 'wb') as fp:
            pickle.dump(self, fp)

    def load_results(self, model_dir: Optional[str] = None):
        model_dir = self.model_dir if model_dir is None else model_dir
        filepath = os.path.join(model_dir, 'model.pkl')
        with open(filepath, 'rb') as fp:
            obj = pickle.load(fp)
        return obj

    def predict_next_week(self) -> pd.DataFrame:
        df = pd.read_csv(self.link_to_data, parse_dates=['gameday'])
        df = df[df['gameday'] > (pd.Timestamp(self.TODAY) - datetime.timedelta(days=self.window))]
        df = self.engineer_features(df)
        df_ = df[
            # Next week of NFL
            df['gameday'].between(pd.Timestamp(self.TODAY), pd.Timestamp(self.TODAY) + datetime.timedelta(days=10))
            |
            # Keep this SF game as a test case
            (df['game_id'] == '2023_07_SF_MIN')
        ].copy()
        df_ = df_[(~df_['money_line'].isna() & ~df_['spread_line'].isna()) | (df_['game_id'] == '2023_07_SF_MIN')]

        # margin of victory for home-team is like a spread for away team
        df_['predicted_margin_of_victory_for_home_team'] = self.predict_spread(df_)
        df_['spread_against_spread'] = df_['predicted_margin_of_victory_for_home_team'] - df_['spread_line']
        print(df_[[
            'game_id',
            'gameday',
            'spread_actual',
            'spread_line',
            'money_line',
            'predicted_margin_of_victory_for_home_team',
            'spread_against_spread'
        ]])
        df_.to_csv(os.path.join(os.getcwd(), 'data', 'df_test.csv'), index=False)
        return df_
