import os
import time
import datetime
from typing import Tuple
import pandas as pd

from sports_bettors.analytics.model.policy import Policy
from config import Config


class Model(object):

    def __init__(self):
        self.models = {
            'nfl': {
                'spread': Policy(league='nfl', response='spread').load_results(),
                'over': Policy(league='nfl', response='over').load_results()
            },
            'college_football': {
                'spread': Policy(league='college_football', response='spread').load_results(),
                'over': Policy(league='college_football', response='over').load_results(),
            }
        }
        self.save_dir = os.path.join(os.getcwd(), 'data', 'predictions')

    def predict_next_week(self):
        for league, models in self.models.items():
            df_out = []
            for response, model in models.items():

                if league == 'nfl':
                    df = pd.read_csv(model.link_to_data, parse_dates=['gameday'])
                    df = df[df['gameday'] > (pd.Timestamp(model.TODAY) - datetime.timedelta(days=model.window))]
                    df = model._add_metrics(df)
                elif league == 'college_football':
                    df = model._download_college_football(predict=True)
                    df = model._add_metrics(df)
                else:
                    raise NotImplementedError(league)

                # Engineer features from raw
                df = model.wrangle(df)

                # Filter for predictions
                test_games = ['2023_07_SF_MIN', 'COLLEGE_TEST_GAME']
                df = df[
                    # Next week of League
                    df['gameday'].between(pd.Timestamp(model.TODAY), pd.Timestamp(model.TODAY) + datetime.timedelta(days=10))
                    |
                    # Keep this SF game as a test case
                    df['game_id'].isin(test_games)
                ].copy()

                # Filter for bad features
                for feature in model.features:
                    df = df[~df[feature].isna()]
                # Get preds as expected "actual" spread / total from model
                df['preds'] = model.predict(df)
                # Get diff from odds-line
                df['preds_against_line'] = df['preds'] - df[model.line_col]
                # Label bets based on human-derived thresholds
                df['Bet'] = df['preds_against_line'].apply(model.apply_policy)
                # df['Bet'] = df.apply(lambda r: Config.label_bet(league, response, r['preds_against_line']), axis=1)
                df['Bet_type'] = response
                df_out.append(df)
            df_out = pd.concat(df_out)
            # Pivot on bet-type
            df_out = df_out[df_out['Bet_type'] == 'spread'].\
                drop('Bet_type', axis=1).\
                rename(columns={'preds': 'spread_adj', 'preds_against_line': 'model_vs_spread', 'Bet': 'Spread_Bet'}).\
                merge(
                    df_out[df_out['Bet_type'] == 'over'].\
                      drop('Bet_type', axis=1).\
                      rename(columns={'preds': 'over_adj', 'preds_against_line': 'model_vs_over', 'Bet': 'Over_Bet'})[
                            ['game_id', 'gameday', 'over_adj', 'model_vs_over', 'Over_Bet']
                      ], on=['game_id', 'gameday'], how='left'
                )
            print(df_out[[
                'game_id',
                'gameday',
                'money_line',
                'spread_line',
                'spread_adj',
                'model_vs_spread',
                'Spread_Bet',
                'total_line',
                'over_adj',
                'model_vs_over',
                'Over_Bet',
               ]
            ])
            # Save results
            save_dir = os.path.join(os.getcwd(), 'data', 'predictions', league)
            fn = f'df_{int(time.time())}.csv'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            df_out.to_csv(os.path.join(save_dir, fn), index=False)
            self.format_for_consumption(df_out, league)

    def format_for_consumption(self, df: pd.DataFrame, league: str):
        """
        Save it to a nice excel file
        """
        df['away_team'] = df['game_id'].apply(lambda s: s.split('_')[-2])
        df['home_team'] = df['game_id'].apply(lambda s: s.split('_')[-1])
        df['gameday'] = df['gameday'].dt.date.astype(str)
        for col in ['money_line', 'spread_adj', 'over_adj']:
            df[col] = df[col].round(2)
        df['Spread_from_Model_for_Away_Team'] = df.\
            apply(
            lambda r: -r['spread_adj'] if r['away_is_favorite'] == 1 else r['spread_adj'],
            axis=1
        )
        df_x = df[[
            'game_id',
            'gameday',
            'home_team',
            'away_team',
            'away_is_favorite',
            'money_line',
            'spread_line',
            'Spread_from_Model_for_Away_Team',
            'Spread_Bet',
            'total_line',
            'over_adj',
            'Over_Bet',
        ]].rename(
            columns={
                'spread_line': 'Spread_from_Vegas_for_Away_Team',
                'total_line': 'Over_Line_from_Vegas',
                'over_adj': 'Over_Line_from_Model',
                'money_line': 'payout_per_dollar_bet_on_away_team_moneyline'
            }
        )
        df_x['away_is_favorite'] = df_x['away_is_favorite'].replace({1: 'Yes', 0: 'No'})
        week_no = datetime.datetime.now().isocalendar()[1]
        save_dir = os.path.join(self.save_dir, league, str(week_no))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df_x.to_excel(os.path.join(save_dir, f'{league}_predictions.xlsx'), index=False)
