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

    def validate(self, df_: Optional[pd.DataFrame] = None, df_val: Optional[pd.DataFrame] = None,
                         df: Optional[pd.DataFrame] = None, run_shap: bool = False):
        if any([df_ is None, df_val is None, df is None]):
            df_, df_val, df = self.fit_transform()

        # Get predictions
        df_['preds'] = self.predict(df_)
        df_val['preds'] = self.predict(df_val)
        df['preds'] = self.predict(df)

        # Preds vs Response
        with PdfPages(os.path.join(self.save_dir, 'validate_spreads.pdf')) as pdf:
            plt.figure()
            plt.text(0.04, 0.95, f'League: {self.league}, response: {self.response}')
            plt.text(0.04, 0.90, f'Train N: {df_.shape[0]}')
            plt.text(0.04, 0.85, f'Val N: {df_val.shape[0]}')
            plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
            pdf.savefig()
            plt.close()

            if self.response == 'spread':
                bins = np.linspace(-50, 50, 21)
            elif self.response == 'over':
                bins = np.linspace(10, 90, 17)
            else:
                raise NotImplementedError(self.response)

            plt.figure()
            plt.hist(df_val['preds'], alpha=0.5, label='preds', bins=bins)
            plt.hist(df_val[self.response_col], alpha=0.5, label='Actuals', bins=bins)
            plt.text(bins[4], 10, f'Preds: {df_val["preds"].mean().round(2)} +/- {df_val["preds"].std().round(2)}')
            plt.text(bins[4], 20, f'Actuals: {df_val[self.response_col].mean().round(2)} +/- {df_val[self.response_col].std().round(2)}')
            plt.xlabel(self.response)
            plt.title('Preds Distribution')
            plt.grid(True)
            plt.legend()
            pdf.savefig()
            plt.close()

            df_val['res'] = df_val['preds'] - df_val[self.response_col]
            df_['res'] = df_['preds'] - df_[self.response_col]
            bins = np.linspace(-50, 50, 21)
            plt.figure()
            plt.hist(df_val['res'], alpha=0.5, label='Test', density=True, bins=bins)
            plt.hist(df_['res'], alpha=0.5, label='Train', density=True, bins=bins)
            plt.text(-40, 0.01, f'Test: {df_val["res"].mean().round(2)} +/- {df_val["res"].std().round(2)}')
            plt.text(-40, 0.02, f'Train: {df_["res"].mean().round(2)} +/- {df_["res"].std().round(2)}')
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
            bins = np.linspace(-50, 50, 21)
            plt.figure()
            plt.hist(df_val['res'], alpha=0.5, label='residuals', bins=bins)
            plt.hist(df_val[self.diff_col], alpha=0.5, label='spread-diff', bins=bins)
            plt.text(-40, 10, f'Residuals: {df_val["res"].mean().round(2)} +/- {df_val["res"].std().round(2)}')
            plt.text(-40, 20, f'Line-Diff: {df_val[self.diff_col].mean().round(2)} +/- {df_val[self.diff_col].std().round(2)}')
            plt.grid(True)
            plt.legend()
            plt.title('Model vs. Line')
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
            # preds-c is the amount above the line the favorite is expected to win by
            # or how much over the over it was expected to be
            # in each case, the actual should be above the line so they are positively correlated
            df['preds_c'] = df['preds'] - df[self.line_col]
            # Response is if the favorite won by more than the line, or the over hit
            # (positively correlated with preds-c)
            classifier_response = 'classifier_response'
            df[classifier_response] = (df[self.response_col] > df[self.line_col]).astype(int)
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

            # Precision recall by test / train
            precision, recall, thresholds = precision_recall_curve(df_[classifier_response], df_['preds_c'])
            plt.figure()
            plt.plot(thresholds, precision[1:], label='precision')
            plt.plot(thresholds, recall[1:], label='recall')
            plt.hlines(1-df_[classifier_response].mean(), min(thresholds), max(thresholds), color='black')
            plt.hlines(df_[classifier_response].mean(), min(thresholds), max(thresholds), color='gray')
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
            plt.hlines(1-df_val[classifier_response].mean(), min(thresholds), max(thresholds), color='black')
            plt.hlines(df_val[classifier_response].mean(), min(thresholds), max(thresholds), color='gray')
            plt.grid(True)
            plt.title('Test')
            pdf.savefig()
            plt.close()

            # Win-Rate by month
            df_plot = df_val[['gameday', classifier_response]].copy()
            df_plot['month'] = df_plot['gameday'].dt.month
            df_plot = df_plot.groupby('month').agg(win_rate=(classifier_response, 'mean')).reset_index()
            plt.figure()
            plt.bar(df_plot['month'], df_plot['win_rate'])
            plt.title('Bias check')
            plt.xlabel('Month')
            plt.ylabel('Win Rate')
            pdf.savefig()
            plt.close()

            # Betting Guide data
            records, n_total = [], df_val.shape[0]
            if self.response == 'spread':
                thresholds = np.linspace(-10, 10, 41)
                labels = 'Favorite', 'Underdog'
                round_interval = 0.5
            elif self.response == 'over':
                thresholds = np.linspace(-20, 20, 41)
                labels = 'Over', 'Under'
                round_interval = 1.
            else:
                raise NotImplementedError(self.response)

            for threshold in thresholds:
                record = {
                    'threshold': threshold,
                    # preds_c is the amount we expect the favorite to win by or the over to hit
                    'fraction_games': df_val[df_val['preds_c'] > threshold].shape[0] / n_total,
                    'fraction_games_interval': df_val[df_val['preds_c'].between(threshold - round_interval, threshold + round_interval)].shape[0] / n_total,
                    'win_rate': df_val[df_val['preds_c'] > threshold][classifier_response].mean(),
                    'win_rate_interval': df_val[df_val['preds_c'].between(threshold - round_interval, threshold + round_interval)][classifier_response].mean(),
                    'team': labels[0]
                }
                records.append(record)
                record = {
                    'threshold': threshold,
                    'fraction_games': df_val[df_val['preds_c'] < threshold].shape[0] / n_total,
                    'fraction_games_interval': df_val[df_val['preds_c'].between(threshold - round_interval, threshold + round_interval)].shape[0] / n_total,
                    'win_rate': (1 - df_val[(df_val['preds_c']) < threshold][classifier_response]).mean(),
                    'win_rate_interval': (1 - df_val[df_val['preds_c'].between(threshold - round_interval, threshold + round_interval)][classifier_response]).mean(),
                    'team': labels[1]
                }
                records.append(record)
            df_plot = pd.DataFrame.from_records(records)

            # Cumulative
            plt.figure()
            for team, df_ in df_plot.groupby('team'):
                df_ = df_[df_['fraction_games'] > 0.05]
                plt.plot(df_['threshold'], df_['win_rate'], label=team)
            if self.response == 'spread':
                plt.xlabel('Predicted Spread on the Vegas-Spread')
            plt.legend()
            plt.ylabel('Win Rate - Cumulative')
            plt.hlines(1-df_val[classifier_response].mean(), min(thresholds), max(thresholds), color='black')
            plt.hlines(df_val[classifier_response].mean(), min(thresholds), max(thresholds), color='gray')
            plt.title('Betting Guide (Cumulative)')
            plt.grid(True)
            pdf.savefig()
            plt.close()

            for team, df_ in df_plot.groupby('team'):
                plt.plot(df_['threshold'], df_['fraction_games'], label=team)
            if self.response == 'spread':
                plt.xlabel('Predicted Spread on the Vegas-Spread')
            plt.legend()
            plt.ylabel('Cumulative Fraction of Games with Good Odds (>52.5%)')
            plt.title('Betting Guide: Number of Games (Cumulative)')
            plt.grid(True)
            pdf.savefig()
            plt.close()

            plt.figure()
            for team, df_ in df_plot.groupby('team'):
                df_ = df_[df_['fraction_games'] > 0.01]
                plt.plot(df_['threshold'], df_['win_rate_interval'], label=team)
            if self.response == 'spread':
                plt.xlabel('Predicted Spread on the Vegas-Spread')
            plt.legend()
            plt.ylabel('Win Rate')
            plt.hlines(1 - df_val[classifier_response].mean(), min(thresholds), max(thresholds), color='black')
            plt.hlines(df_val[classifier_response].mean(), min(thresholds), max(thresholds), color='gray')
            plt.title('Betting Guide')
            plt.grid(True)
            pdf.savefig()
            plt.close()

            plt.figure()
            for team, df_ in df_plot.groupby('team'):
                plt.plot(df_['threshold'], df_['fraction_games_interval'], label=team)
            if self.response == 'spread':
                plt.xlabel('Predicted Spread on the Vegas-Spread')
            plt.legend()
            plt.ylabel('Fraction of Games with Good Odds (>52.5%)')
            plt.title('Betting Guide: Number of Games ')
            plt.grid(True)
            pdf.savefig()
            plt.close()
            plt.figure()
