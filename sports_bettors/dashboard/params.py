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
    },
    'feature_creators': {
        'v2': {
            'college_football': {
                'RushOnly': {
                    'rush_yds_x_atms': lambda p: p['rushingYards'] * p['rushingAttempts']
                },
                'PassOnly': {
                    'pass_yds_x_atms': lambda p: p['netPassingYards'] * p['passAttempts']
                },
                'Offense': {
                    'rush_yds_x_atms': lambda p: p['rushingYards'] * p['rushingAttempts'],
                    'pass_yds_x_atms': lambda p: p['netPassingYards'] * p['passAttempts'],
                    'rush_yds_x_pass_yds': lambda p: p['rushingYards'] * p['netPassingYards']
                },
                'OffenseAdv': {
                    'rush_yds_adv_x_pass_yds_adv': lambda p: p['rush_yds_adv'] * p['pass_yds_adv']
                },
                'PointsScored': {}
            },
            'nfl': {
                'RushOnly': {
                    'rush_yds_x_atms': lambda p: p['rushYards'] * p['rushAttempts']
                },
                'PassOnly': {
                    'pass_yds_x_atms': lambda p: p['NetPassYards'] * p['passAttempts']
                },
                'Offense': {
                    'rush_yds_x_atms': lambda p: p['rushYards'] * p['rushAttempts'],
                    'pass_yds_x_atms': lambda p: p['NetPassYards'] * p['passAttempts'],
                    'rush_yds_x_pass_yds': lambda p: p['rushYards'] * p['NetPassYards']
                },
                'OffenseAdv': {
                    'rush_yds_adv_x_pass_yds_adv': lambda p: p['rush_yds_adv'] * p['pass_yds_adv']
                },
                'PointsScored': {}
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
                {'label': 'Points Scored', 'value': 'PointsScored'},
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
                'rushYards': range(50, 300, 5),
                'NetPassYards': range(50, 500, 5),
                'rush_yds_adv': range(-200, 200, 5),
                'pass_yds_adv': range(-300, 300, 5),
                'to_margin': range(-3, 4),
                'total_points': range(10, 100)
            },
            'college_football': {
                'rushingAttempts': range(5, 30),
                'passAttempts': range(5, 30),
                'rushingYards': range(50, 300, 5),
                'netPassingYards': range(50, 500, 5),
                'rush_yds_adv': range(-200, 200, 5),
                'pass_yds_adv': range(-300, 300, 5),
                'to_margin': range(-3, 4),
                'total_points': range(10, 100)
            }
        },
        'response-ranges': {
            'nfl': {
                'WinMargin': range(1, 28),
                'LossMargin': range(1, 28),
                'Margin': range(-28, 28),
                'TotalPoints': range(10, 100)
            },
            'college_football': {
                'WinMargin': range(1, 28),
                'LossMargin': range(1, 28),
                'Margin': range(-28, 28),
                'TotalPoints': range(10, 100)
            }
        }
    }
}
