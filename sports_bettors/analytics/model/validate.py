import os
import pickle
from typing import Optional
import numpy as np
import pandas as pd
import datetime
import shap
from tqdm import tqdm
from scipy.stats import binomtest

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

from sports_bettors.analytics.model.model import Model
from config import logger, Config


class Validate(Model):
    # Response col label
    classifier_response = 'classifier_response'
    # In policy, correct for biases that aren't 50/50
    bias_correction = True

    def __init__(self, league: str = 'nfl', response: str = 'spread', overwrite: bool = False):
        super().__init__(league=league, response=response, overwrite=overwrite)
        if self.response == 'spread':
            self.policy = {
                'left': {
                    'name': 'Underdog',
                    'threshold': None
                },
                'right': {
                    'name': 'Favorite',
                    'threshold': None
                }
            }
        elif self.response == 'over':
            self.policy = {
                'left': {
                    'name': 'Under',
                    'threshold': None
                },
                'right': {
                    'name': 'Over',
                    'threshold': None
                }
            }

    def discover_policy(self, df: pd.DataFrame, pdf: PdfPages):
        logger.info('Discovering best policy')
        df = df[['game_id', 'preds_c', self.classifier_response]]
        thresholds = np.linspace(-10, 10, 41)
        result_records = []
        for left_threshold in tqdm(thresholds):
            for right_threshold in thresholds[thresholds > left_threshold]:
                for game_record in df.to_dict(orient='records'):
                    result_record = {
                        'game_id': game_record['game_id'],
                        'preds_c': game_record['preds_c'],
                        self.classifier_response: game_record[self.classifier_response],
                        'left_threshold': left_threshold,
                        'right_threshold': right_threshold
                    }
                    if game_record['preds_c'] <= left_threshold:
                        result_record['left_bet'] = True
                        result_record['left_correct'] = game_record[self.classifier_response] == 0
                    else:
                        result_record['left_bet'] = False
                        result_record['left_correct'] = None
                    if game_record['preds_c'] >= right_threshold:
                        result_record['right_bet'] = True
                        result_record['right_correct'] = game_record[self.classifier_response] == 1
                    else:
                        result_record['right_bet'] = False
                        result_record['right_correct'] = None
                    result_records.append(result_record)
        df_policy = pd.DataFrame.from_records(result_records)
        df_policy = df_policy.groupby(['left_threshold', 'right_threshold']).\
            agg(
                num_left_bet=('left_bet', 'sum'),
                num_right_bet=('right_bet', 'sum'),
                num_left_wins=('left_correct', 'sum'),
                num_right_wins=('right_correct', 'sum'),
                num_games=('game_id', 'nunique')
            ).reset_index()
        # Totals for the policy
        df_policy['num_bets'] = df_policy['num_left_bet'] + df_policy['num_right_bet']
        df_policy['num_wins'] = df_policy['num_left_wins'] + df_policy['num_right_wins']
        # Left and right wins can be subject to bias that I don't want my policy dependent on generalizing
        # To correct, we'll calculate a win-rate for each side...
        df_policy['left_win_rate'] = df_policy.\
            apply(lambda r: r['num_left_wins'] / r['num_left_bet'] if r['num_left_bet'] > 0 else 0, axis=1)
        df_policy['right_win_rate'] = df_policy.\
            apply(lambda r: r['num_right_wins'] / r['num_right_bet'] if r['num_right_bet'] > 0 else 0, axis=1)
        # ... then calculate the bias from 50% for each side
        left_bias = (1 - df[self.classifier_response]).mean() - 0.5
        right_bias = df[self.classifier_response].mean() - 0.5
        # Assume that these wins are due to luck / bias and won't generalize and subtract / add to the observed num_wins
        df_policy['num_left_wins_eff'] = (df_policy['left_win_rate'] - left_bias).clip(0, 1) * df_policy['num_left_bet']
        df_policy['num_right_wins_eff'] = (df_policy['right_win_rate'] - right_bias).clip(0, 1) * df_policy['num_right_bet']

        # Optionally apply bias correction
        if self.bias_correction:
            df_policy['num_wins'] = df_policy['num_left_wins_eff'] + df_policy['num_right_wins_eff']
        else:
            df_policy['num_wins'] = df_policy['num_left_wins'] + df_policy['num_right_wins']

        # Have to have some bets and a win
        df_policy = df_policy[(df_policy['num_wins'] > 0) & (df_policy['num_bets'] > 0)]
        df_policy['win_rate'] = df_policy['num_wins'] / df_policy['num_bets']

        # P-value assumes a coin-flip is the baseline probability of getting a spread right but we will
        # be more conservative and set it to 52.5 to account for the vigorish
        # Alternatively you could compare to your own intuition or some policy like, "Always bet right"
        df_policy['p_value'] = df_policy.\
            apply(lambda r: binomtest(int(r['num_wins']), int(r['num_bets']), p=0.525, alternative='greater').pvalue, axis=1)
        # Expected return with a conservative edge case of 0.5
        df_policy['expected_win_rate'] = (df_policy['win_rate'] * (1 - df_policy['p_value']) + 0.5 * df_policy['p_value'])
        # Assume payout for a win is ~0.909 to account for vigorish
        df_policy['expected_return'] = 0.909 * df_policy['expected_win_rate'] * df_policy['num_bets'] - 1 * (1 - df_policy['expected_win_rate']) * df_policy['num_bets']

        # Save policy-check work
        df_policy.to_csv(os.path.join(self.save_dir, 'df_policy_check.csv'), index=False)

        # save results to policy
        self.policy['left']['threshold'] = df_policy[df_policy['expected_return'] == df_policy['expected_return'].max()]['left_threshold'].iloc[0]
        self.policy['right']['threshold'] = df_policy[df_policy['expected_return'] == df_policy['expected_return'].max()]['right_threshold'].iloc[0]

    def apply_policy(self, p: float) -> str:
        # Manual thresholds, use that
        if Config.manual_policy[self.league][self.response]:
            return Config.label_bet(self.league, self.response, p)

        if p < self.policy['left']['threshold']:
            return self.policy['left']['name']
        elif p > self.policy['right']['threshold']:
            return self.policy['right']['name']
        else:
            return 'No Bet'

    def assess_policy(self, df: pd.DataFrame) -> pd.Series:
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
        return df.apply(lambda r: _bet_result(r['Bet'], r[self.classifier_response]), axis=1)

    def validate(self, df_: Optional[pd.DataFrame] = None, df_val: Optional[pd.DataFrame] = None,
                         df: Optional[pd.DataFrame] = None, run_shap: bool = False):
        if any([df_ is None, df_val is None, df is None]):
            df_, df_val, df = self.fit_transform()

        # Get predictions
        df_['preds'] = self.predict(df_)
        df_val['preds'] = self.predict(df_val)
        df['preds'] = self.predict(df)

        # Preds vs Response
        with PdfPages(os.path.join(self.save_dir, 'validate.pdf')) as pdf:
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
            for label, df_plot in {'train': df_, 'test': df_val}.items():
                fpr, tpr, thresholds = roc_curve(df_plot[self.classifier_response], df_plot['preds_c'])
                auc = roc_auc_score(df_plot[self.classifier_response], df_plot['preds_c'])
                plt.figure()
                plt.plot(fpr, tpr)
                plt.text(0.2, 0.9, f'AUC: {auc:.3f}\nn={df_plot.shape[0]}')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.title(label)
                plt.grid(True)
                pdf.savefig()
                plt.close()

            # Precision recall by test / train
            for label, df_plot in {'train': df_, 'test': df_val}.items():
                precision, recall, thresholds = precision_recall_curve(df_plot[self.classifier_response], df_plot['preds_c'])
                plt.figure()
                plt.plot(thresholds, precision[1:], label='precision')
                plt.plot(thresholds, recall[1:], label='recall')
                plt.hlines(1-df_plot[self.classifier_response].mean(), min(thresholds), max(thresholds), color='black')
                plt.hlines(df_plot[self.classifier_response].mean(), min(thresholds), max(thresholds), color='gray')
                plt.legend()
                plt.grid(True)
                plt.title(label)
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
            self.discover_policy(df_val, pdf)
            df_policy['Bet'] = df_policy['preds_c'].apply(self.apply_policy)
            df_policy['Bet_result'] = self.assess_policy(df_policy)

            # Note: No Bet is a null so it won't be summed
            df_plot = df_policy.copy()
            num_wins = df_plot['Bet_result'].sum()
            num_losses = (1 - df_plot['Bet_result']).sum()
            num_bets = df_plot[~df_plot['Bet_result'].isna()].shape[0]
            win_rate = round(num_wins / num_bets, 3) if num_bets > 0 else None
            p_value = binomtest(int(num_wins), int(num_bets), p=0.5, alternative='greater').pvalue
            plt.text(0.04, 0.95, f'League: {self.league}, response: {self.response}')
            plt.text(0.04, 0.90, 'Time-Frame: {}'.format('past_year'))
            plt.text(0.04, 0.85, f'Record: {int(num_wins)}-{int(num_losses)}')
            plt.text(0.04, 0.80, f'Win Percentage: {win_rate} (p={round(p_value, 3)})')

            # Season so far
            # This season so far (Note: maximum this can be is 126 days, so we'll just do the last 180 days
            df_policy = df_policy[df_policy['gameday'] > pd.Timestamp(self.TODAY) - datetime.timedelta(days=180)].copy()
            df_policy['Bet_result'] = self.assess_policy(df_policy)
            # Note: No Bet is a null so it won't be summed
            df_plot = df_policy.copy()
            num_wins = df_plot['Bet_result'].sum()
            num_losses = (1 - df_plot['Bet_result']).sum()
            num_bets = df_plot[~df_plot['Bet_result'].isna()].shape[0]
            win_rate = round(num_wins / num_bets, 3) if num_bets > 0 else None
            p_value = binomtest(int(num_wins), int(num_bets), p=0.5, alternative='greater').pvalue
            plt.text(0.04, 0.70, f'League: {self.league}, response: {self.response}')
            plt.text(0.04, 0.65, 'Time-Frame: {}'.format('This Season so Far'))
            plt.text(0.04, 0.60, f'Record: {int(num_wins)}-{int(num_losses)}')
            plt.text(0.04, 0.55, f'Win Percentage: {win_rate} (p={round(p_value, 3)})')

            # Past Week
            df_policy = df_policy[df_policy['gameday'] > pd.Timestamp(self.TODAY) - datetime.timedelta(days=8)].copy()
            df_policy['Bet_result'] = self.assess_policy(df_policy)
            # Note: No Bet is a null so it won't be summed
            df_plot = df_policy.copy()
            num_wins = df_plot['Bet_result'].sum()
            num_losses = (1 - df_plot['Bet_result']).sum()
            num_bets = df_plot[~df_plot['Bet_result'].isna()].shape[0]
            win_rate = round(num_wins / num_bets, 3) if num_bets > 0 else None
            p_value = binomtest(int(num_wins), int(num_bets), p=0.5, alternative='greater').pvalue
            plt.text(0.04, 0.45, f'League: {self.league}, response: {self.response}')
            plt.text(0.04, 0.40, 'Time-Frame: {}'.format('Past 7 Days'))
            plt.text(0.04, 0.35, f'Record: {int(num_wins)}-{int(num_losses)}')
            plt.text(0.04, 0.30, f'Win Percentage: {win_rate} (p={round(p_value, 3)})')

            plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
            pdf.savefig()
            plt.close()
