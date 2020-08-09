import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly.express as px

import pandas as pd

from sports_bettors.dashboard.params import params, utils
from sports_bettors.dashboard.history import populate as history_populate
from sports_bettors.dashboard.results import populate as results_populate

from config import Config


def add_sb_dash(server, routes_pathname_prefix: str = '/api/dash/sportsbettors/'):
    """
    Add a sports-bettors dashboard over a flask app at the provided endpoint
    """
    dashapp = dash.Dash(
        __name__,
        routes_pathname_prefix=routes_pathname_prefix,
        server=server
    )

    dashapp.layout = html.Div(children=[
        html.H1('Hi From Dash (sports bettors)'),
        dcc.Dropdown(id='league', options=params[Config.dashboard_version]['league-opts'], value='college_football'),
        dcc.Dropdown(id='team', style=utils['no_show']),
        dcc.Dropdown(id='opponent', style=utils['no_show']),

        # History
        html.Div(id='history-data', style=utils['no_show'], children=pd.DataFrame().to_json()),
        dcc.Dropdown(id='history-x', style=utils['no_show']),
        dcc.Dropdown(id='history-y', style=utils['no_show']),
        html.Button('Update History', id='update-history-data', n_clicks=0),
        dcc.Graph(id='history-fig'),

        # Results
        html.Div(id='results-data', style=utils['no_show'], children=pd.DataFrame().to_json()),
        dcc.Dropdown(id='feature-sets', style=utils['no_show']),
        html.Button('Update Results', id='update-results-data', n_clicks=0),
        dcc.Graph(id='win-fig'),
        dcc.Graph(id='margin-fig')
    ])

    # Drop down population
    @dashapp.callback(
        [
            Output('team', 'options'),
            Output('team', 'style'),
            Output('opponent', 'options'),
            Output('opponent', 'style'),
            Output('feature-sets', 'options'),
            Output('feature-sets', 'style')
        ],
        [
            Input('league', 'value')
        ]
    )
    def config_dropdowns(league):
        team_opts = params[Config.dashboard_version]['team-opts'][league]
        feature_set_opts = params[Config.dashboard_version]['feature-sets-opts'][league]
        return team_opts, utils['show'], team_opts, utils['show'], feature_set_opts, utils['show']

    # Populate with history
    @dashapp.callback(
        [
            Output('history-data', 'children'),
            Output('history-x', 'options'),
            Output('history-y', 'options')
        ],
        [
            Input('update-history-data', 'n_clicks'),
            Input('league', 'value'),
        ],
        [
            State('team', 'value'),
            State('opponent', 'value')
        ]
    )
    def history_data(trigger, league, team, opponent):
        df, x_opts, y_opts = history_populate(league, team, opponent)
        return df.to_json(), x_opts, y_opts

    # Make the figure
    @dashapp.callback(
        [
            Output('history-fig', 'figure'),
            Output('history-x', 'style'),
            Output('history-y', 'style')
         ],
        [
            Input('history-data', 'children'),
            Input('history-x', 'value'),
            Input('history-y', 'value')
        ]
    )
    def history_figures(df, x, y):
        df = pd.read_json(df, orient='records')
        if df.shape[0] == 0:
            return utils['empty_figure'], utils['no_show'], utils['no_show']
        x = df.columns[0] if not x else x
        y = df.columns[0] if not y else y
        fig = px.scatter(df, x=x, y=y, color='Winner')
        return fig, utils['show'], utils['show']

    # Populate with results
    @dashapp.callback(
        Output('results-data', 'children'),
        [
            Input('update-results-data', 'n_clicks')
        ],
        [
            State('league', 'value'),
            State('feature-sets', 'value'),
            State('team', 'value'),
            State('opponent', 'value')
        ]
    )
    def results_data(trigger, league, feature_set, team, opponent):
        df = results_populate(league=league, feature_set=feature_set, team=team, opponent=opponent)
        return df.to_json()

    # Make the figure
    @dashapp.callback(
        [
            Output('win-fig', 'figure'),
            Output('margin-fig', 'figure')
        ],
        [
            Input('results-data', 'children')
        ]
    )
    def results_figures(df):
        df = pd.read_json(df, orient='records')
        if df.shape[0] == 0:
            return utils['empty_figure'], utils['empty_figure']
        fig = px.line(df, x='total_points', y='Win')
        return fig, fig

    # Load historical match-up data (if any)

    # Select a match-up (team_a and team_b)
    # Plot results for team_a as 'team' and team_b as 'opponent' (inverted)

    # On load; calculate default data set. The change in these parameters will be triggered by a button:
    #   League: college-football;
    #   Conditions: PointsScored;
    #   Team: Iowa
    #   Opponent: Wisconsin
    #   Output: Probability or Log-odds (We might be able to return this and toggle it reactively)

    # Store data in 3 hidden divs
    # for team in teams:
    #   for total_points_in_game in range(10, 100):
    #       calculate probability of winning with error bars (Main Figure) for each team
    #           Fields: Team, TotalPoints, LB, E, UB
    #       calculate probability of winning with error bars by 0 -> 21 for each team (Figure that depends on slice)
    #           Fields: Team, TotalPoints, WinMargin, LB, E, UB
    #       calculate probability of losing with error bars by 0 -> 21 for each team (Figure that depends on slice)
    #           Fields: Team, TotalPoints, LossMargin, LB, E, UB
    #       calculate probability of margin with error bars by -21 -> 21 for each team (Figure that depends on slice)

    # Normalize all probabilities of winning so that the sum = 1
    # Combine margins for an expected margin
    # Combine Win/Loss margins

    return dashapp.server
