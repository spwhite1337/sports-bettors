import os
import pandas as pd

from config import Config

college_teams = set(pd.read_csv(os.path.join(Config.DATA_DIR, 'sports_bettors', 'curated', 'college_football',
                                             'df_curated.csv'))['team'])
nfl_teams = set(pd.read_csv(os.path.join(Config.DATA_DIR, 'sports_bettors', 'curated', 'nfl',
                                         'df_curated.csv'))['team'])

utils = {
    'empty_figure': {
        "layout": {
            "xaxis": {
                "visible": False
            },
            "yaxis": {
                "visible": False
            },
            "annotations": [
                {
                    "text": "No matching data found",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 28
                    }
                }
            ]
        }
    },
    'no_show': {'display': 'none'},
    'show': {'display': 'block'},
    'feature_maps': {
        'v2': {
            'college_football': {
                'Rushing Attempts': 'rushingAttempts',
                'Rushing Yards': 'rushingYards',
                'Passing Attempts': 'passAttempts',
                'Passing Yards': 'netPassingYards',
                'Turnover Margin': 'to_margin',
                'Rushing Yards Advantage': 'rush_yds_adv',
                'Passing Yards Advantage': 'pass_yds_adv',
                'Points Scored': 'total_points',
            },
            'nfl': {
                'Rushing Attempts': 'rushAttempts',
                'Rushing Yards': 'rushYards',
                'Passing Attempts': 'passAttempts',
                'Passing Yards': 'NetPassYards',
                'Turnover Margin': 'to_margin',
                'Rushing Yards Advantage': 'rush_yds_adv',
                'Passing Yards Advantage': 'pass_yds_adv',
                'Points Scored': 'total_points',
            }
        }
    }
}

params = {
    'v2': {
        'league-opts': [
            {'label': 'NFL', 'value': 'nfl'},
            {'label': 'College Football', 'value': 'college_football'}
        ],
        'team-opts': {
            'nfl': [{'label': team, 'value': team} for team in nfl_teams],
            'college_football': [{'label': team, 'value': team} for team in college_teams]
        },
        'feature-sets-opts': {
            'nfl': [
                {'label': 'Rushing', 'value': 'RushOnly'},
                {'label': 'Passing', 'value': 'PassOnly'},
                {'label': 'Offense', 'value': 'Offense'},
                {'label': 'Offense (Advantage)', 'value': 'OffenseAdv'},
                # {'label': 'Points Scored', 'value': 'PointsScored'},
            ],
            'college_football': [
                {'label': 'Rushing', 'value': 'RushOnly'},
                {'label': 'Passing', 'value': 'PassOnly'},
                {'label': 'Offense', 'value': 'Offense'},
                {'label': 'Offense (Advantage)', 'value': 'OffenseAdv'},
                {'label': 'Points Scored', 'value': 'PointsScored'}
            ]
        },
        'variable-opts': {
            'nfl': {
                'RushOnly': [
                    {'label': 'Rushing Attempts', 'value': 'rushAttempts'},
                    {'label': 'Rushing Yards', 'value': 'rushYards'}
                ],
                'PassOnly': [
                    {'label': 'Passing Attempts', 'value': 'passAttempts'},
                    {'label': 'Passing Yards', 'value': 'NetPassYards'}
                ],
                'Offense': [
                    {'label': 'Rushing Yards', 'value': 'rushYards'},
                    {'label': 'Rushing Attempts', 'value': 'rushAttempts'},
                    {'label': 'Passing Yards', 'value': 'NetPassYards'},
                    {'label': 'Passing Attempts', 'value': 'passAttempts'}
                ],
                'OffenseAdv': [
                    {'label': 'Rushing Yards Advantage', 'value': 'rush_yds_adv'},
                    {'label': 'Passing Yards Advantage', 'value': 'pass_yds_adv'},
                    {'label': 'Turnover Margin', 'value': 'to_margin'}
                ],
                'PointsScored': [
                    {'label': 'Points Scored', 'value': 'total_points'},
                ]
            },
            'college_football': {
                'RushOnly': [
                    {'label': 'Rushing Attempts', 'value': 'rushingAttempts'},
                    {'label': 'Rushing Yards', 'value': 'rushingYards'},
                ],
                'PassOnly': [
                    {'label': 'Passing Attempts', 'value': 'passAttempts'},
                    {'label': 'Passing Yards', 'value': 'netPassingYards'}
                ],
                'Offense': [
                    {'label': 'Rushing Attempts', 'value': 'rushingAttempts'},
                    {'label': 'Rushing Yards', 'value': 'rushingYards'},
                    {'label': 'Passing Attempts', 'value': 'passAttempts'},
                    {'label': 'Passing Yards', 'value': 'netPassingYards'}
                ],
                'OffenseAdv': [
                    {'label': 'Rushing Yards Advantage', 'value': 'rush_yds_adv'},
                    {'label': 'Passing Yards Advantage', 'value': 'pass_yds_adv'},
                    {'label': 'Turnover Margin', 'value': 'to_margin'}
                ],
                'PointsScored': [
                    {'label': 'Points Scored', 'value': 'total_points'},
                ]
            }
        },
        'variable-ranges': {
            'nfl': {
                'rushAttempts': range(5, 30),
                'passAttempts': range(5, 30),
                'rushYards': range(50, 300),
                'NetPassYards': range(50, 500),
                'rush_yds_adv': range(-200, 200),
                'pass_yds_adv': range(-300, 300),
                'to_margin': range(-3, 4),
                'total_points': range(10, 100)
            },
            'college_football': {
                'rushingAttempts': range(5, 30),
                'passAttempts': range(5, 30),
                'rushingYards': range(50, 300),
                'netPassingYards': range(50, 500),
                'rush_yds_adv': range(-200, 200),
                'pass_yds_adv': range(-300, 300),
                'to_margin': range(-3, 4),
                'total_points': range(10, 100)
            }
        }
    }
}
