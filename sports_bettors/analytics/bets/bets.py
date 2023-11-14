from typing import Optional
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import binomtest
from datetime import datetime


class Bets(object):
    TODAY = datetime.strftime(datetime.today(), '%Y-%m-%d')

    def __init__(self):
        self.save_dir = os.path.join(os.getcwd(), 'docs', 'bets')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

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
            os.path.join(os.getcwd(), 'data', 'sports_bettors', 'raw_archive', 'betting', 'bets.csv'),
            parse_dates=['Date']
        )
        df = df[[
            'Date', 'Money', 'Bet_Type', 'Number', 'Odds', 'Supporting_Team',
            'Opposing_Team', 'League', 'Result', 'Amount', 'Model_Agree'
        ]]
        df = df[df['Odds'].between(-130, 300)].copy()
        df['payout'] = df['Odds'].apply(self._calc_payout)
        df['Net_Gain'] = df.apply(self.actual_gain, axis=1)
        return df

    def analyze(self, df: Optional[pd.DataFrame] = None):
        if df is None:
            df = self.etl()

        with PdfPages(os.path.join(self.save_dir, 'bets.pdf')) as pdf:
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

            plt.figure()
            plt.text(0.04, 0.9,
                     f'Overall:\nWin Rate: {win_rate:.3f} at {bets_per_week:0.2f} bets/week '
                     f'for {gain_per_week:0.2f} units/week.'
                     f'\np_value:{p_value:0.2f}')
            plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
            pdf.savefig()
            plt.close()

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

                plt.figure()
                plt.text(0.04, 0.9, f'{league}:\nWin Rate: {win_rate:.3f} at {bets_per_week:0.2f} bets/week '
                      f'for {gain_per_week:0.2f} units/week.'
                      f'\np_value:{p_value:0.2f}')
                plt.text(0.04, 0.5, (df_['Result'].value_counts()))
                plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
                pdf.savefig()
                plt.close()

            # Cumulative gains of bets
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
            pdf.savefig()
            plt.close()

            df['Model_Agree'] = df['Model_Agree'].fillna('Not Used')
            for (league, bet_type), df_ in df.groupby(['League', 'Bet_Type']):
                plt.figure()
                for model_agree, df_plot in df_.groupby('Model_Agree'):
                    df_plot['cumsum'] = df_plot['Net_Gain'].cumsum()
                    df_plot['Bet_no'] = df_plot.reset_index(drop=True).index
                    plt.plot(df_plot['Bet_no'], df_plot['cumsum'], label=model_agree)
                plt.title(f'{league}: {bet_type}')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                pdf.savefig()
                plt.close()

            # Curate categories a bit
            df['Model_Agree'] = df['Model_Agree'].replace({
                'Against Max Return': 'Not Used',
                'Against Min Risk': 'Not Used',
                'No Bet': 'Not Used',
                'Max Return': 'Model Used',
                'Min Risk': 'Model Used'
            })
            df['Bet_Type'] = df['Bet_Type'].replace({'Under': 'Over'})

            for (league, bet_type), df_ in df.groupby(['League', 'Bet_Type']):
                plt.figure()
                for model_agree, df_plot in df_.groupby('Model_Agree'):
                    num_wins = df_plot[df_plot['Result'] == 'Won'].shape[0]
                    num_losses = df_plot[df_plot['Result'] == 'Lost'].shape[0]
                    num_ties = df_plot[df_plot['Result'] == 'Push'].shape[0]
                    df_plot['cumsum'] = df_plot['Net_Gain'].cumsum()
                    df_plot['Bet_no'] = df_plot.reset_index(drop=True).index
                    plt.plot(df_plot['Bet_no'], df_plot['cumsum'],
                             label=str(model_agree) + f' ({num_wins}-{num_losses}-{num_ties})')
                plt.title(f'{league}: {bet_type} (Curated)')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                pdf.savefig()
                plt.close()
