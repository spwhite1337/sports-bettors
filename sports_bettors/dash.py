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
        html.Div(id='selectors', children=[
            dcc.Dropdown(id='league', options=params[Config.version]['league-opts'], value='college_football'),
            dcc.Dropdown(id='team', style=utils['no_show']),
            dcc.Dropdown(id='opponent', style=utils['no_show']),
        ]),

        # History
        html.Div(id='history', children=[
            html.Div(id='history-data', style=utils['no_show'], children=pd.DataFrame().to_json()),
            dcc.Dropdown(id='history-x', style=utils['no_show']),
            dcc.Dropdown(id='history-y', style=utils['no_show']),
            html.Button('Update History', id='update-history-data', n_clicks=0),
            dcc.Graph(id='history-fig'),
        ]),

        # Results
        html.Div(id='results', children=[
            html.Div(id='results-data', style=utils['no_show'], children=pd.DataFrame().to_json()),
            dcc.Dropdown(id='feature-sets', style=utils['no_show']),
            dcc.Dropdown(id='variable', style=utils['no_show']),
            html.Button('Update Results', id='update-results-data', n_clicks=0),
            dcc.Graph(id='win-fig'),
            dcc.Graph(id='margin-fig')
        ]),
    ])

    # Drop down population
    @dashapp.callback(
        [
            Output('team', 'options'),
            Output('team', 'style'),
            Output('opponent', 'options'),
            Output('opponent', 'style'),
            Output('feature-sets', 'options'),
            Output('feature-sets', 'style'),
        ],
        [
            Input('league', 'value')
        ]
    )
    def config_dropdowns(league):
        team_opts = params[Config.version]['team-opts'][league]
        feature_set_opts = params[Config.version]['feature-sets-opts'][league]
        return team_opts, utils['show'], team_opts, utils['show'], feature_set_opts, utils['show']

    @dashapp.callback(
        [
            Output('variable', 'options'),
            Output('variable', 'style')
        ],
        [
            Input('league', 'value'),
            Input('feature-sets', 'value'),
        ]
    )
    def config_variables(league, feature_set):
        if (league is None) or (feature_set is None):
            return [], utils['no_show']
        variable_opts = params[Config.version]['variable-opts'][league][feature_set]
        return variable_opts, utils['show']

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
            State('opponent', 'value'),
            State('variable', 'value')
        ]
    )
    def results_data(trigger, league, feature_set, team, opponent, variable):
        df = results_populate(league=league, feature_set=feature_set, team=team, opponent=opponent, variable=variable)
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

    # Define a 'variable' for each feature_set, then rest of the conditions are parameters
    # Calculate WinProbability / UB / LB for a range of "variable" values at the fixed parameters
    # For each slice of variable, parameters -> calculate probability of winning by various margins,
    #   loss margins, win-margins
    # Result will be 4 dfs:
    #   - Win Probability as variable is changed
    #       - [variable, parameters, WinProb, WinProbUB, WinProbLB, team]
    #   - Margin Likelihood as variable is changed
    #       - [variable, parameters, margin, CumProb, CumProbLB, CumProbUB, team]
    #   - WinMargin Likelihood
    #   - LossMargin Likelihood

    # Normalize probabilities across teams
    #   - Combine margins for an expected margin
    #   - Combine Win/Loss margins

    return dashapp.server
