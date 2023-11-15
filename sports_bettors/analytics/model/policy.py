import os
import datetime
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import binomtest

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sports_bettors.analytics.model.validate import Validate
from config import logger, Config


class Policy(Validate):

    bias_correction = True

    def __init__(self, league: str = 'nfl', response: str = 'spread', overwrite: bool = False):
        super().__init__(league=league, response=response, overwrite=overwrite)
        self.policies = {
            'spread': {
                'max_return': {
                    'left': {
                        'name': 'Underdog',
                        'threshold': None
                    },
                    'right': {
                        'name': 'Favorite',
                        'threshold': None
                    }
                },
                'top_decile': {
                    'left': {
                        'name': 'Underdog',
                        'threshold': None
                    },
                    'right': {
                        'name': 'Favorite',
                        'threshold': None
                    }
                },
                'top_quartile': {
                    'left': {
                        'name': 'Underdog',
                        'threshold': None
                    },
                    'right': {
                        'name': 'Favorite',
                        'threshold': None
                    }
                },
                'top_half': {
                    'left': {
                        'name': 'Underdog',
                        'threshold': None
                    },
                    'right': {
                        'name': 'Favorite',
                        'threshold': None
                    }
                },
                'min_risk': {
                    'left': {
                        'name': 'Underdog',
                        'threshold': None
                    },
                    'right': {
                        'name': 'Favorite',
                        'threshold': None
                    }
                }
            },
            'over': {
                'max_return': {
                    'left': {
                        'name': 'Under',
                        'threshold': None
                    },
                    'right': {
                        'name': 'Over',
                        'threshold': None
                    }
                },
                'top_decile': {
                    'left': {
                        'name': 'Under',
                        'threshold': None
                    },
                    'right': {
                        'name': 'Over',
                        'threshold': None
                    }
                },
                'top_quartile': {
                    'left': {
                        'name': 'Under',
                        'threshold': None
                    },
                    'right': {
                        'name': 'Over',
                        'threshold': None
                    }
                },
                'top_half': {
                    'left': {
                        'name': 'Under',
                        'threshold': None
                    },
                    'right': {
                        'name': 'Over',
                        'threshold': None
                    }
                },
                'min_risk': {
                    'left': {
                        'name': 'Under',
                        'threshold': None
                    },
                    'right': {
                        'name': 'Over',
                        'threshold': None
                    }
                },
            }
        }[response]

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
        df_policy = df_policy.groupby(['left_threshold', 'right_threshold']). \
            agg(
            num_left_bet=('left_bet', 'sum'),
            num_right_bet=('right_bet', 'sum'),
            num_left_wins=('left_correct', 'sum'),
            num_right_wins=('right_correct', 'sum'),
            num_games=('game_id', 'nunique')
        ).reset_index().assign(total_num_games=df_policy['game_id'].nunique())
        # Totals for the policy
        df_policy['num_bets'] = df_policy['num_left_bet'] + df_policy['num_right_bet']
        df_policy['num_wins'] = df_policy['num_left_wins'] + df_policy['num_right_wins']
        # Left and right wins can be subject to bias that I don't want my policy dependent on generalizing
        # To correct, we'll calculate a win-rate for each side...
        df_policy['left_win_rate'] = df_policy. \
            apply(lambda r: r['num_left_wins'] / r['num_left_bet'] if r['num_left_bet'] > 0 else 0, axis=1)
        df_policy['right_win_rate'] = df_policy. \
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

        # P-value assumes a coin-flip is the baseline probability of getting a spread right
        # Alternatively you could compare to your own intuition or some policy like, "Always bet right"
        df_policy['p_value'] = df_policy. \
            apply(lambda r: binomtest(int(r['num_wins']), int(r['num_bets']), p=0.50, alternative='greater').pvalue,
                  axis=1)
        # Expected return with a conservative edge case of 0.5
        df_policy['expected_win_rate'] = (df_policy['win_rate'] * (1 - df_policy['p_value']) + 0.5 * df_policy['p_value'])
        df_policy['expected_return'] = 1.0 * df_policy['expected_win_rate'] * df_policy['num_bets'] - 1 * (
                1 - df_policy['expected_win_rate']) * df_policy['num_bets']

        # Save policy-check work
        df_policy.to_csv(os.path.join(self.save_dir, f'df_policy_check.csv'), index=False)

        # save results to policy for max-expected-return
        self.policies['max_return']['left']['threshold'] = \
            df_policy[df_policy['expected_return'] == df_policy['expected_return'].max()]['left_threshold'].iloc[0]
        self.policies['max_return']['right']['threshold'] = \
            df_policy[df_policy['expected_return'] == df_policy['expected_return'].max()]['right_threshold'].iloc[0]

        # Save results to policy for min_risk
        # Must have a positive return with a decent p-value
        df_min_risk = df_policy[(df_policy['expected_return'] > 0) & (df_policy['p_value'] <= 0.1)]
        df_min_risk = df_min_risk[df_min_risk['expected_win_rate'] == df_min_risk['expected_win_rate'].max()]
        if df_min_risk.shape[0] == 0:
            self.policies['min_risk']['left']['threshold'] = None
            self.policies['min_risk']['right']['threshold'] = None
        else:
            self.policies['min_risk']['left']['threshold'] = df_min_risk['left_threshold'].iloc[0]
            self.policies['min_risk']['right']['threshold'] = df_min_risk['right_threshold'].iloc[0]

        # Save results for thresholded cutoffs
        for t_name, t in [
            ('top_decile', 0.1),
            ('top_quartile', 0.25),
            ('top_half', 0.5),
        ]:
            if t_name in self.policies.keys():
                df_t = df_policy.copy()
                df_t['diff'] = (df_t['num_bets'] / df_t['num_games'] - t).abs()
                # +/- 10% of threshold
                df_t = df_t[df_t['diff'].between(df_t['diff'].min() * 0.9, df_t['diff'].min() * 1.1)]
                df_t = df_t[df_t['expected_return'] == df_t['expected_return'].max()]
                self.policies[t_name]['left']['threshold'] = df_t['left_threshold'].iloc[0]
                self.policies[t_name]['right']['threshold'] = df_t['right_threshold'].iloc[0]

        # Make graph with thresholds
        df_plot = []
        for policy, p_params in self.policies.items():
            for direction, d_params in p_params.items():
                record = {
                    'policy': policy,
                    'threshold': d_params['name'],
                    'value': d_params['threshold'],
                    'direction': direction
                }
                df_plot.append(record)
        df_plot = pd.DataFrame.from_records(df_plot)
        plt.figure()
        for direction, df_plot_ in df_plot.groupby('direction'):
            plt.bar(df_plot_['policy'], df_plot_['value'], label=direction, alpha=0.5)
        plt.legend()
        plt.grid(True)
        plt.xlabel('Direction')
        plt.ylabel('Threshold Amount')
        pdf.savefig()
        plt.close()

    def apply_policy(self, p: float, policy: str) -> str:
        l_threshold = self.policies[policy]['left']['threshold']
        r_threshold = self.policies[policy]['right']['threshold']
        if l_threshold is not None:
            if p < l_threshold:
                return self.policies[policy]['left']['name']
        if r_threshold is not None:
            if p > r_threshold:
                return self.policies[policy]['right']['name']
        return 'No Bet'

    def assess_policy(self, df: pd.DataFrame, policy: str) -> pd.Series:
        left_name = self.policies[policy]['left']['name']
        right_name = self.policies[policy]['right']['name']

        # How well does the bet match the result?
        def _bet_result(bet: str, result: float) -> Optional[float]:
            # Got it right
            if (result == 1 and bet == right_name) or (result == 0 and bet == left_name):
                return 1.
            # Got it wrong
            elif (result == 0 and bet == right_name) or (result == 1 and bet == left_name):
                return 0.
            # Doesn't meet either decision threshold
            elif bet == 'No Bet':
                return None
            else:
                return None

        return df.apply(lambda r: _bet_result(r['Bet'], r[self.classifier_response]), axis=1)

    def validate(self,
                 df_: Optional[pd.DataFrame] = None,
                 df_val: Optional[pd.DataFrame] = None,
                 df: Optional[pd.DataFrame] = None,
                 run_shap: bool = False):
        if any([df_ is None, df_val is None, df is None]):
            df_, df_val, df = self.fit_transform(val=True)

        with PdfPages(os.path.join(self.save_dir, 'validate.pdf')) as pdf:
            # Validate model before moving on to policies
            df_val = self.validate_model(pdf, df_=df_, df_val=df_val, df=df, run_shap=run_shap)

            self.discover_policy(df_val, pdf)
            for policy, policy_params in self.policies.items():
                df_policy = df_val[['game_id', 'gameday', 'preds_c', self.classifier_response]].copy()
                df_policy['Bet'] = df_policy['preds_c'].apply(lambda p: self.apply_policy(p, policy))
                df_policy['Bet_result'] = self.assess_policy(df_policy, policy)

                # Get win-rate and records for a few time-frames
                # Whole year
                # Note: No Bet is a null so it won't be summed
                df_plot = df_policy.copy()
                num_wins = df_plot['Bet_result'].sum()
                num_losses = (1 - df_plot['Bet_result']).sum()
                num_bets = df_plot[~df_plot['Bet_result'].isna()].shape[0]
                win_rate = round(num_wins / num_bets, 3) if num_bets > 0 else None
                if any([num_wins == 0, num_bets == 0]):
                    p_value = np.nan
                else:
                    p_value = binomtest(int(num_wins), int(num_bets), p=0.5, alternative='greater').pvalue

                df_policy['week'] = df_policy['gameday'].dt.year * 52 + df_policy['gameday'].dt.isocalendar().week
                df_policy['yes_bet'] = (~df_policy['Bet_result'].isna()).astype(int)
                df_plot = df_policy[df_policy['yes_bet'] == 1].copy()
                if df_plot.shape[0] > 0:
                    # Weekly win-loss trends
                    df_plot = df_plot.\
                        groupby('week').\
                        agg(win_rate=('Bet_result', 'mean'), num_bets=('yes_bet', 'sum')).\
                        reset_index()
                    plt.figure()
                    plt.bar(df_plot['week'], df_plot['win_rate'])
                    plt.hlines(0.5, df_plot['week'].min(), df_plot['week'].max())
                    plt.grid(True)
                    plt.xlabel('Week No')
                    plt.ylabel('Win Rate')
                    plt.title(f'{self.league}, {self.response}: {policy}')
                    pdf.savefig()
                    plt.close()

                    # Number of bets by week
                    plt.figure()
                    plt.bar(df_plot['week'], df_plot['num_bets'])
                    plt.grid(True)
                    plt.xlabel('Week No')
                    plt.ylabel('Number of Bets')
                    plt.title(f'{self.league}, {self.response}: {policy}')
                    pdf.savefig()
                    plt.close()

                plt.figure()
                plt.text(0.04, 0.95, f'League: {self.league}, response: {self.response}, policy: {policy}')
                plt.text(0.04, 0.90, 'Time-Frame: {}'.format('past_year'))
                plt.text(0.04, 0.85, f'Record: {int(num_wins)}-{int(num_losses)} '
                                     f'(Bet Percentage: {round(100 * int(num_bets) / df_policy.shape[0], 1)}%)')
                plt.text(0.04, 0.80, f'Win Percentage: {win_rate} (p={round(p_value, 3)}) ')

                # Season so far
                # This season so far (Note: maximum this can be is 126 days, so we'll just do the last 180 days
                df_policy = df_policy[df_policy['gameday'] > pd.Timestamp(self.TODAY) - datetime.timedelta(days=180)].copy()
                df_policy['Bet_result'] = self.assess_policy(df_policy, policy)
                # Note: No Bet is a null so it won't be summed
                df_plot = df_policy.copy()
                num_wins = df_plot['Bet_result'].sum()
                num_losses = (1 - df_plot['Bet_result']).sum()
                num_bets = df_plot[~df_plot['Bet_result'].isna()].shape[0]
                win_rate = round(num_wins / num_bets, 3) if num_bets > 0 else None
                if any([num_wins == 0, num_bets == 0]):
                    p_value = np.nan
                else:
                    p_value = binomtest(int(num_wins), int(num_bets), p=0.5, alternative='greater').pvalue
                plt.text(0.04, 0.70, f'League: {self.league}, response: {self.response}, policy: {policy}')
                plt.text(0.04, 0.65, 'Time-Frame: {}'.format('This Season so Far'))
                plt.text(0.04, 0.60, f'Record: {int(num_wins)}-{int(num_losses)} '
                                     f'(Bet Percentage: {round(100 * int(num_bets) / df_policy.shape[0], 1)}%)')
                plt.text(0.04, 0.55, f'Win Percentage: {win_rate} (p={round(p_value, 3)})')

                # Past Week
                df_policy = df_policy[df_policy['gameday'] > pd.Timestamp(self.TODAY) - datetime.timedelta(days=8)].copy()
                df_policy['Bet_result'] = self.assess_policy(df_policy, policy)
                # Note: No Bet is a null so it won't be summed
                df_plot = df_policy.copy()
                num_wins = df_plot['Bet_result'].sum()
                num_losses = (1 - df_plot['Bet_result']).sum()
                num_bets = df_plot[~df_plot['Bet_result'].isna()].shape[0]
                win_rate = round(num_wins / num_bets, 3) if num_bets > 0 else None
                if any([num_wins == 0, num_bets == 0]):
                    p_value = np.nan
                else:
                    p_value = binomtest(int(num_wins), int(num_bets), p=0.5, alternative='greater').pvalue
                plt.text(0.04, 0.45, f'League: {self.league}, response: {self.response}, policy: {policy}')
                plt.text(0.04, 0.40, 'Time-Frame: {}'.format('Past 7 Days'))
                plt.text(0.04, 0.35, f'Record: {int(num_wins)}-{int(num_losses)} '
                                     f'(Bet Percentage: {round(100 * int(num_bets) / df_policy.shape[0], 1)}%)')
                plt.text(0.04, 0.30, f'Win Percentage: {win_rate} (p={round(p_value, 3)})')
                plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
                pdf.savefig()
                plt.close()
