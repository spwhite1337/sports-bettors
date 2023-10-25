import os
import time
import datetime
from typing import Tuple
import pandas as pd

from sports_bettors.analytics.model.validate import Validate
from config import Config


class Model(object):

    def __init__(self):
        self.models = {
            'nfl': {
                'spread': Validate(league='nfl', response='spread').load_results(),
                'over': Validate(league='nfl', response='over').load_results()
            },
            'college_football': {
                'spread': Validate(league='college_football', response='spread').load_results(),
                'over': Validate(league='college_football', response='over').load_results(),
            }
        }

    def predict_next_week(self):
        for league, models in self.models.items():
            df_out = []
            for response, model in models.items():

                if league == 'nfl':
                    df = pd.read_csv(model.link_to_data, parse_dates=['gameday'])
                    df = df[df['gameday'] > (pd.Timestamp(model.TODAY) - datetime.timedelta(days=model.window))]
                elif league == 'college_football':
                    df = model._download_college_football(predict=True)
                else:
                    raise NotImplementedError(league)

                # Engineer features from raw
                df = model.calcs(df)
                df = model.engineer_features(df)
                df = model.label_teams(df)

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
                    df = df[~df[feature].isna() | (df['game_id'].isin(test_games))]

                # Margin of victory for home-team is like a spread for away team
                df['preds'] = model.predict(df)
                df['preds_against_line'] = df['preds'] - df[model.line_col]
                # Label bets
                df['Bet'] = df.apply(lambda r: Config.label_bet(league, response, r['preds_against_line']), axis=1)
                df['Bet_type'] = response

                # # Print results
                # print(df[[
                #     'game_id',
                #     'gameday',
                #     'spread_line',
                #     'total_line',
                #     'preds',
                #     'preds_against_line',
                #     'Bet',
                #     'Bet_type'
                # ]])
                # print('Bet Counts')
                # print(df['Bet'].value_counts())

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
                'total_line'
                'spread_adj',
                'model_vs_spread',
                'Spread_Bet',
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
