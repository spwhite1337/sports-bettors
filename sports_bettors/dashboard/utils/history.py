import os
from typing import Tuple
import pandas as pd

from config import Config


def populate(league, team, opponent) -> Tuple[pd.DataFrame, list, list]:
    if league == 'college_football':
        df = pd.read_csv(os.path.join(Config.DATA_DIR, 'sports_bettors', 'curated', league, 'df_curated.csv'))
        df['Winner'] = df['team'].where(df['points'] > df['opp_points'], df['opponent'])
        df = df[(df['team'] == team) & (df['opponent'] == opponent)]
        options = [{'label': col, 'value': col} for col in df.columns]
        return df, options, options
    elif league == 'nfl':
        df = pd.read_csv(os.path.join(Config.DATA_DIR, 'sports_bettors', 'curated', league, 'df_curated.csv'))
        df['Winner'] = df['team'].where(df['points'] > df['opp_points'], df['opponent'])
        df = df[(df['team'] == team) & (df['opponent'] == opponent)]
        options = [{'label': col, 'value': col} for col in df.columns]
        return df, options, options
    else:
        return pd.DataFrame(), [], []
