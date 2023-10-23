from typing import Optional
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from datetime import datetime


class Eda(object):
    TODAY = datetime.strftime(datetime.today(), '%Y-%m-%d')

    @staticmethod
    def _calc_payout(odds: float) -> float:
        if odds < 0:
            return 100 / abs(odds)
        else:
            return abs(odds) / 100

    @staticmethod
    def actual_gain(row) -> Optional[float]:
        if row['Result'] == 'Won':
            return row['Money'] * row['payout']
        elif row['Result'] == 'Lost':
            return - row['Money']
        elif row['Result'] == 'Push':
            return 0.
        else:
            return None

    def etl(self) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(
                os.path.dirname(os.path.dirname(os.getcwd())),
                'data', 'sports_bettors', 'raw_archive', 'betting', 'bets.csv'
            ), parse_dates=['Date']
        )
        df = df[[
            'Date', 'Money', 'Bet Type', 'Number', 'Odds', 'Supporting Team',
            'Opposing Team', 'League', 'Result', 'Amount'
        ]]
        df = df[df['Odds'].between(-130, 300)].copy()
        df['payout'] = df['Odds'].apply(self._calc_payout)
        df['Net_Gain'] = df.apply(self.actual_gain, axis=1)
        return df

    def analyze(self, df: Optional[pd.DataFrame] = None):
        if df is None:
            df = self.etl()

        # Binomial test
        p_value = binomtest(
            df[df['Result'] == 'Won'].shape[0],
            df[df['Result'].isin(['Won', 'Lost'])].shape[0],
            p=0.5,  # Assume probability of 0.5 on spreads without vigorish
            alternative='greater'
        ).pvalue
        win_rate = (df['Result'] == 'Won').sum() / (df['Result'].isin(['Won', 'Lost']).sum())
        bets_per_week = (df.shape[0] / (df['Date'].max() - df['Date'].min()).days) * 7
        payout = df['payout'].mean()
        gain_per_week = (payout * win_rate - (1 - win_rate)) * bets_per_week

        print(f'Overall:\nWin Rate: {win_rate:.3f} at {bets_per_week:0.2f} bets/week '
              f'for {gain_per_week:0.2f} units/week.'
              f'\np_value:{p_value:0.2f}')
        print('')

        for league, df_ in df.groupby('League'):
            df_ = df[df['League'] == league]
            win_rate = (df_['Result'] == 'Won').sum() / (df_['Result'].isin(['Won', 'Lost']).sum())
            bets_per_week = (df_.shape[0] / (df['Date'].max() - df['Date'].min()).days) * 7
            payout = df_['payout'].mean()
            gain_per_week = (payout * win_rate - (1 - win_rate)) * bets_per_week
            p_value = binomtest(
                df_[df_['Result'] == 'Won'].shape[0],
                df_[df_['Result'].isin(['Won', 'Lost'])].shape[0],
                p=0.5,
                alternative='greater'
            ).pvalue
            print(f'{league}:\nWin Rate: {win_rate:.3f} at {bets_per_week:0.2f} bets/week '
                  f'for {gain_per_week:0.2f} units/week.'
                  f'\np_value:{p_value:0.2f}')
            print(df_['Result'].value_counts())
            print('')

        df['cumsum'] = df.groupby('League')['Net_Gain'].cumsum()
        df_plot = pd.concat([
            df[df['League'] == 'College'][['Date', 'cumsum', 'League']].reset_index(drop=True),
            df[df['League'] == 'NFL'][['Date', 'cumsum', 'League']].reset_index(drop=True),
        ])
        df_plot['Bet_no'] = df_plot.index

        plt.figure()
        for league, df_plot_ in df_plot.groupby('League'):
            plt.plot(df_plot_['Bet_no'], df_plot_['cumsum'], label=league)
        plt.legend()
        plt.grid(True)
        plt.xlabel('Bet No.')
        plt.ylabel('Cumulative Gain')
        plt.title('Cumulative Gain on Bets by League')
        plt.show()
