from typing import Optional, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Eda(object):
    link_to_data = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
    min_date = '2017-06-01'

    @staticmethod
    def moneyline_to_prob(ml: float) -> float:
        odds = 100 / abs(ml) if ml < 0 else abs(ml) / 100
        return 1 - odds / (1 + odds)

    @staticmethod
    def _calc_metrics(df: pd.DataFrame) -> pd.DataFrame:
        # Metrics
        df['spread_actual'] = df['home_score'] - df['away_score']
        df['spread_diff'] = df['away_score'] + df['spread_line'] - df['home_score']
        df['total_actual'] = df['away_score'] + df['home_score']
        df['off_spread'] = (df['spread_actual'] - df['spread_line'])
        df['off_total'] = (df['total_actual'] - df['total_line'])
        return df

    @staticmethod
    def _result_spread_category(row: Dict) -> str:
        if (row['spread_line'] < 0) and (row['spread_diff'] > 0):
            return 'Favorite Covered'
        elif (row['spread_line'] > 0) and (row['spread_diff'] < 0):
            return 'Favorite Covered'
        elif row['spread_diff'] == 0:
            return 'Push'
        else:
            return 'Underdog Covered'

    @staticmethod
    def _result_total_category(row: Dict) -> str:
        if row['off_total'] < 0:
            return 'Under'
        elif row['off_total'] == 0:
            return 'Push'
        else:
            return 'Over'

    def etl(self) -> pd.DataFrame:
        df = pd.read_csv(self.link_to_data, parse_dates=['gameday'])
        df = df[
            # Drop planned games
            ~df['away_score'].isna()
            &
            # Time filter
            (df['gameday'] > pd.Timestamp(self.min_date))
            &
            # Only regular season
            (df['game_type'] == 'REG')
        ]
        return df

    def spread_accuracy(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is None:
            df = self.etl()
        df = self._calc_metrics(df)
        df['spread_result'] = df.apply(self._result_spread_category, axis=1)
        df['total_result'] = df.apply(self._result_total_category, axis=1)
        return df

    def moneyline_accuracy(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is None:
            df = self.etl()
        df['away_win_prob'] = df['away_moneyline'].apply(self.moneyline_to_prob)
        df['win_prob_bucket'] = df['away_win_prob'].round(1)
        df['away_win'] = (df['away_score'] > df['home_score']).astype(int)
        df = df.\
            groupby('win_prob_bucket'). \
            agg(
                win_actual=('away_win', 'mean'),
                num_wins=('away_win', 'sum'),
                n=('away_win', 'count'),
                win_prob_mean=('away_win_prob', 'mean'),
            ).reset_index()
        df['p_wins'] = df['win_prob_mean'] * df['n']
        df['freq'] = df['n'] / df['n'].sum()

        df = df[['win_prob_bucket', 'win_prob_mean', 'win_actual', 'num_wins', 'p_wins', 'n', 'freq']]
        df['gross_gain'] = df['num_wins'] * (1 - df['win_prob_mean']) / (df['win_prob_mean'])
        df['gross_loss'] = df['n'] - df['num_wins']
        df['net_gain'] = df['gross_gain'] - df['gross_loss']
        df['net_gain_per_bet'] = df['net_gain'] / df['n']
        df['net_gain'].sum() / df['n'].sum()
        return df

    def analyze(self):
        df = self.etl()

        # Moneyline accuracy
        df_ml = self.moneyline_accuracy(df)
        plt.figure()
        df_ml['win_prob_bucket'] = df_ml['win_prob_bucket'].astype(str)
        plt.bar(df_ml['win_prob_bucket'], df_ml['win_actual'])
        plt.xlabel('Predict Win Probability')
        plt.ylabel('Actual Win Probability')
        plt.grid(True)
        plt.show()

        df__ = df_ml[df_ml['win_prob_bucket'] != '0.0']
        plt.bar(df__['win_prob_bucket'], df__['net_gain_per_bet'])
        plt.xlabel('Predict Win Probability')
        plt.ylabel('Net Gain per Bet')
        plt.grid(True)
        plt.show()

        # Spread / Total Accuracy
        df_s = self.spread_accuracy(df)
        bins = np.linspace(-50, 50, 21)

        plt.figure()
        plt.hist(df_s['spread_actual'], alpha=0.5, label='Actual', bins=bins)
        plt.hist(df_s['spread_diff'], label='spread_corrected', alpha=0.5, bins=bins)
        plt.text(-20, 80, '{} +/- {}'.format(
            round(df_s['spread_actual'].mean(), 2),
            round(df_s['spread_actual'].std(), 2)
        ))
        plt.text(20, 80, '{} +/- {}'.format(
            round(df_s['spread_diff'].mean(), 2),
            round(df_s['spread_diff'].std(), 2)
        ))
        plt.grid(True)
        plt.vlines(df_s['spread_diff'].mean(), 0, 100)
        plt.vlines(df_s['spread_diff'].median(), 0, 100)
        plt.title('Margin of Victory for away team')
        plt.legend()
        plt.show()

        plt.figure()
        plt.hist(df_s['off_total'], alpha=0.5)
        plt.text(-20, 80, '{} +/- {}, Median: {}'.format(
            round(df_s['off_total'].mean(), 2),
            round(df_s['off_total'].std(), 2),
            round(df_s['off_total'].median(), 2)
        ))
        plt.vlines(df_s['off_total'].mean(), 0, 100)
        plt.vlines(df_s['off_total'].median(), 0, 100)
        plt.title('Points Total Against Line')
        plt.grid(True)
        plt.show()

        plt.figure()
        print('{}% of Games are <= 3 points ATS'.format(
            round((df_s['spread_diff'].abs() <= 3).sum() / df_s.shape[0] * 100, 2)))
        print('{}% of Games are <= 7 points ATS'.format(
            round((df_s['spread_diff'].abs() <= 7).sum() / df_s.shape[0] * 100, 2)))
        plt.hist(df_s['spread_diff'].abs(), cumulative=True, density=True, bins=np.linspace(0, 28, 29))
        plt.xlabel('Margin of Victory Against the Spread')
        plt.ylabel('Fraction of Games')
        plt.grid(True)
        plt.show()
