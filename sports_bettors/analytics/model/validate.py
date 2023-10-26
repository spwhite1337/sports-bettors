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
from config import logger, Config


class Validate(Model):
    # Response col label
    classifier_response = 'classifier_response'

    def assess_policy(self, df: pd.DataFrame) -> pd.DataFrame:
        # How well does the bet match the result?
        def _bet_result(bet: str, result: float) -> Optional[float]:
            if self.response == 'spread':
                if (result == 1 and bet == 'Favorite') or (result == 0 and bet == 'Underdog'):
                    return 1.
                elif (result == 0 and bet == 'Favorite') or (result == 1 and bet == 'Underdog'):
                    return 0.
                elif bet == 'No Bet':
                    return None
                else:
                    return None
            elif self.response == 'over':
                if (result == 1 and bet == 'Over') or (result == 0 and bet == 'Under'):
                    return 1.
                elif (result == 0 and bet == 'Over') or (result == 1 and bet == 'Under'):
                    return 0.
                elif bet == 'No Bet':
                    return None
                else:
                    return None
        df['Bet'] = df['preds_c'].apply(lambda p: Config.label_bet(self.league, self.response, p))
        df['Bet_result'] = df.apply(lambda r: _bet_result(r['Bet'], r[self.classifier_response]), axis=1)
        return df

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
            df[self.classifier_response] = (df[self.response_col] > df[self.line_col]).astype(int)
            df_ = df[df['gameday'] < (pd.Timestamp(self.TODAY) - pd.Timedelta(days=self.val_window))].copy()
            df_val = df[df['gameday'] > (pd.Timestamp(self.TODAY) - pd.Timedelta(days=self.val_window))].copy()

            # ROC
            fpr, tpr, thresholds = roc_curve(df_[self.classifier_response], df_['preds_c'])
            auc = roc_auc_score(df_[self.classifier_response], df_['preds_c'])
            plt.figure()
            plt.plot(fpr, tpr)
            plt.text(0.2, 0.9, f'AUC: {auc:.3f}\nn={df_val.shape[0]}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title('Train')
            plt.grid(True)
            pdf.savefig()
            plt.close()

            fpr, tpr, thresholds = roc_curve(df_val[self.classifier_response], df_val['preds_c'])
            auc = roc_auc_score(df_val[self.classifier_response], df_val['preds_c'])
            plt.figure()
            plt.plot(fpr, tpr)
            plt.text(0.2, 0.9, f'AUC: {auc:.3f}\nn={df_val.shape[0]}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.grid(True)
            plt.title('Test')
            pdf.savefig()
            plt.close()

            # Precision recall by test / train
            precision, recall, thresholds = precision_recall_curve(df_[self.classifier_response], df_['preds_c'])
            plt.figure()
            plt.plot(thresholds, precision[1:], label='precision')
            plt.plot(thresholds, recall[1:], label='recall')
            plt.hlines(1-df_[self.classifier_response].mean(), min(thresholds), max(thresholds), color='black')
            plt.hlines(df_[self.classifier_response].mean(), min(thresholds), max(thresholds), color='gray')
            plt.legend()
            plt.grid(True)
            plt.title('Train')
            pdf.savefig()
            plt.close()

            precision, recall, thresholds = precision_recall_curve(df_val[self.classifier_response], df_val['preds_c'])
            plt.figure()
            plt.plot(thresholds, precision[1:], label='precision')
            plt.plot(thresholds, recall[1:], label='recall')
            plt.legend()
            plt.hlines(1-df_val[self.classifier_response].mean(), min(thresholds), max(thresholds), color='black')
            plt.hlines(df_val[self.classifier_response].mean(), min(thresholds), max(thresholds), color='gray')
            plt.grid(True)
            plt.title('Test')
            pdf.savefig()
            plt.close()

            # Win-Rate by month
            df_plot = df_val[['gameday', self.classifier_response]].copy()
            df_plot['month'] = df_plot['gameday'].dt.month
            df_plot = df_plot.groupby('month').agg(win_rate=(self.classifier_response, 'mean')).reset_index()
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
                    'n_games': df_val[df_val['preds_c'] > threshold].shape[0],
                    'fraction_games_interval': df_val[df_val['preds_c'].between(threshold - round_interval, threshold + round_interval)].shape[0] / n_total,
                    'n_games_interval': df_val[df_val['preds_c'].between(threshold - round_interval, threshold + round_interval)].shape[0],
                    'win_rate': df_val[df_val['preds_c'] > threshold][self.classifier_response].mean(),
                    'win_rate_interval': df_val[df_val['preds_c'].between(threshold - round_interval, threshold + round_interval)][self.classifier_response].mean(),
                    'team': labels[0]
                }
                records.append(record)
                record = {
                    'threshold': threshold,
                    'fraction_games': df_val[df_val['preds_c'] < threshold].shape[0] / n_total,
                    'n_games': df_val[df_val['preds_c'] < threshold].shape[0],
                    'fraction_games_interval': df_val[df_val['preds_c'].between(threshold - round_interval, threshold + round_interval)].shape[0] / n_total,
                    'n_games_interval': df_val[df_val['preds_c'].between(threshold - round_interval, threshold + round_interval)].shape[0],
                    'win_rate': (1 - df_val[(df_val['preds_c']) < threshold][self.classifier_response]).mean(),
                    'win_rate_interval': (1 - df_val[df_val['preds_c'].between(threshold - round_interval, threshold + round_interval)][self.classifier_response]).mean(),
                    'team': labels[1]
                }
                records.append(record)
            df_plot = pd.DataFrame.from_records(records)

            # Cumulative
            plt.figure()
            for team, df_ in df_plot.groupby('team'):
                df_ = df_[df_['n_games'] > 2]
                plt.plot(df_['threshold'], df_['win_rate'], label=team)
            if self.response == 'spread':
                plt.xlabel('Predicted Spread on the Vegas-Spread')
            elif self.response == 'over':
                plt.xlabel('Predicted Spread on the Over')
            plt.legend()
            plt.ylabel('Win Rate - Cumulative')
            plt.hlines(1-df_val[self.classifier_response].mean(), min(thresholds), max(thresholds), color='black')
            plt.hlines(df_val[self.classifier_response].mean(), min(thresholds), max(thresholds), color='gray')
            plt.title('Betting Guide (Cumulative)')
            plt.grid(True)
            pdf.savefig()
            plt.close()

            for team, df_ in df_plot.groupby('team'):
                plt.plot(df_['threshold'], df_['n_games'], label=team)
            if self.response == 'spread':
                plt.xlabel('Predicted Spread on the Vegas-Spread')
            elif self.response == 'over':
                plt.xlabel('Predicted Spread on the Over')
            plt.legend()
            plt.ylabel('Cumulative Fraction of Games with Good Odds (>52.5%)')
            plt.title('Betting Guide: Number of Games (Cumulative)')
            plt.grid(True)
            pdf.savefig()
            plt.close()

            plt.figure()
            for team, df_ in df_plot.groupby('team'):
                df_ = df_[df_['n_games_interval'] > 2]
                plt.plot(df_['threshold'], df_['win_rate_interval'], label=team)
            if self.response == 'spread':
                plt.xlabel('Predicted Spread on the Vegas-Spread')
            elif self.response == 'over':
                plt.xlabel('Predicted Spread on the Over')
            plt.legend()
            plt.ylabel('Win Rate')
            plt.hlines(1 - df_val[self.classifier_response].mean(), min(thresholds), max(thresholds), color='black')
            plt.hlines(df_val[self.classifier_response].mean(), min(thresholds), max(thresholds), color='gray')
            plt.title('Betting Guide')
            plt.grid(True)
            pdf.savefig()
            plt.close()

            plt.figure()
            for team, df_ in df_plot.groupby('team'):
                plt.plot(df_['threshold'], df_['n_games_interval'], label=team)
            if self.response == 'spread':
                plt.xlabel('Predicted Spread on the Vegas-Spread')
            elif self.response == 'over':
                plt.xlabel('Predicted Spread on the Over')
            plt.legend()
            plt.ylabel('Number of Games')
            plt.title('Betting Guide: Number of Games ')
            plt.grid(True)
            pdf.savefig()
            plt.close()
            plt.figure()

            # Get win-rate and records for a few time-frames
            plt.figure()

            # Whole year
            df_policy = df_val[['game_id', 'gameday', 'preds_c', self.classifier_response]].copy()
            df_plot = self.assess_policy(df_policy)
            # Note: No Bet is a null so it won't be summed
            num_wins = df_plot['Bet_result'].sum()
            num_losses = (1 - df_plot['Bet_result']).sum()
            num_bets = df_plot[~df_plot['Bet_result'].isna()].shape[0]
            win_rate = round(num_wins / num_bets, 3) if num_bets > 0 else None
            plt.text(0.04, 0.95, f'League: {self.league}, response: {self.response}')
            plt.text(0.04, 0.90, 'Time-Frame: {}'.format('past_year'))
            plt.text(0.04, 0.85, f'Record: {int(num_wins)}-{int(num_losses)}')
            plt.text(0.04, 0.80, f'Win Percentage: {win_rate}')

            # Season so far
            # This season so far (Note: maximum this can be is 126 days, so we'll just do the last 180 days
            df_policy = df_policy[df_policy['gameday'] > pd.Timestamp(self.TODAY) - datetime.timedelta(days=180)].copy()
            df_plot = self.assess_policy(df_policy)
            # Note: No Bet is a null so it won't be summed
            num_wins = df_plot['Bet_result'].sum()
            num_losses = (1 - df_plot['Bet_result']).sum()
            num_bets = df_plot[~df_plot['Bet_result'].isna()].shape[0]
            win_rate = round(num_wins / num_bets, 3) if num_bets > 0 else None
            plt.text(0.04, 0.70, f'League: {self.league}, response: {self.response}')
            plt.text(0.04, 0.65, 'Time-Frame: {}'.format('This Season so Far'))
            plt.text(0.04, 0.60, f'Record: {int(num_wins)}-{int(num_losses)}')
            plt.text(0.04, 0.55, f'Win Percentage: {win_rate}')

            # Past Week
            df_policy = df_policy[df_policy['gameday'] > pd.Timestamp(self.TODAY) - datetime.timedelta(days=8)].copy()
            df_plot = self.assess_policy(df_policy)
            # Note: No Bet is a null so it won't be summed
            num_wins = df_plot['Bet_result'].sum()
            num_losses = (1 - df_plot['Bet_result']).sum()
            num_bets = df_plot[~df_plot['Bet_result'].isna()].shape[0]
            win_rate = round(num_wins / num_bets, 3) if num_bets > 0 else None
            plt.text(0.04, 0.45, f'League: {self.league}, response: {self.response}')
            plt.text(0.04, 0.40, 'Time-Frame: {}'.format('Past 7 Days'))
            plt.text(0.04, 0.35, f'Record: {int(num_wins)}-{int(num_losses)}')
            plt.text(0.04, 0.30, f'Win Percentage: {win_rate}')

            plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
            pdf.savefig()
            plt.close()
