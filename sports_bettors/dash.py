import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd

from sports_bettors.dashboard.params import params, utils
from sports_bettors.dashboard.callbacks import ConfigCallbacks, DataCallbacks, PlotCallbacks


from config import Config


def add_sb_dash(server, routes_pathname_prefix: str = '/api/dash/sportsbettors/'):
    """
    Add a sports-bettors dashboard over a flask app at the provided endpoint
    """
    dashapp = dash.Dash(
        __name__,
        routes_pathname_prefix=routes_pathname_prefix,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        server=server
    )

    dashapp.layout = html.Div(children=[
        html.H1('Sports Betting Dashboard'),
        html.Br(), html.Br(),
        html.Div(id='selectors', children=[
            html.H3('Select League, Team, and Opponent'),
            dcc.Dropdown(id='league', options=params[Config.sb_version]['league-opts'], value='college_football',
                         placeholder='Select League'),
            dcc.Dropdown(id='team', style=utils['no_show'], placeholder='Select Team'),
            dcc.Dropdown(id='opponent', style=utils['no_show'], placeholder='Select Opponent'),
        ]),

        # History
        html.Div(id='history', children=[
            html.Div(id='history-data', style=utils['no_show'], children=pd.DataFrame().to_json()),
            html.Br(),
            html.H3('Display Historical Match-up Data (if applicable)'),
            dcc.Dropdown(id='history-x', style=utils['no_show']),
            dcc.Dropdown(id='history-y', style=utils['no_show']),
            dbc.Button('Update History', id='update-history-data', n_clicks=0, color='primary'),
            dcc.Graph(id='history-fig', style=utils['no_show']),
        ]),

        # Results
        html.Div(id='results', children=[
            html.Div(id='results-win-data', style=utils['no_show'], children=pd.DataFrame().to_json()),
            html.Div(id='results-margin-data', style=utils['no_show'], children=pd.DataFrame().to_json()),
            html.Div(id='results-total-points-data', style=utils['no_show'], children=pd.DataFrame().to_json()),
            html.Br(), html.Br(),
            html.H3('Configure Results'),
            html.P(
                """
                First select the model of interest (i.e. RushOnly only controls for Rushing stats. Then select the 
                feature of that selected model you want to treat as a variable for the dashboard (e.g. Display results
                for various values of Rushing Yards). Finally, set the value of the parameters in the model. 
                """
            ),
            html.Br(),
            html.P(
                """
                For example: Select RushOnly for a model that controls for rushing attempts and rushing yards, then
                set your variable as Rushing Yards and fix rushing attempts at 20 to see the results for a model that
                controls for rushing statistics with various values of rushing yards but a fixed value for rushing
                attempts.
                """
            ),
            dcc.Dropdown(id='feature-sets', style=utils['no_show']),
            dcc.Dropdown(id='variable', style=utils['no_show']),
            dbc.Input(id='parameter-1', style=utils['no_show']),
            dbc.Input(id='parameter-2', style=utils['no_show']),
            dbc.Input(id='parameter-3', style=utils['no_show']),
            dbc.Input(id='parameter-4', style=utils['no_show']),
            dbc.Button('Update Results', id='update-results-data', n_clicks=0, color="primary"),
            dcc.Graph(id='win-fig', figure=utils['empty_figure'], style=utils['no_show']),
            dcc.Graph(id='margin-fig', figure=utils['empty_figure'], style=utils['no_show']),
            dcc.Graph(id='total-points-fig', figure=utils['empty_figure'], style=utils['no_show'])
        ]),
    ])

    # Drop down configuration
    @dashapp.callback(
        [
            Output('team', 'options'),
            Output('team', 'style'),
            Output('opponent', 'options'),
            Output('opponent', 'style'),
            Output('feature-sets', 'options'),
            Output('feature-sets', 'style'),
        ],
        [Input('league', 'value')]
    )
    def config_dropdowns(league):
        return ConfigCallbacks.dropdowns(league)

    # Variable Selection
    @dashapp.callback(
        [Output('variable', 'options'), Output('variable', 'style')],
        [Input('league', 'value'), Input('feature-sets', 'value')]
    )
    def config_variables(league, feature_set):
        return ConfigCallbacks.variables(league, feature_set)

    # Parameter Selection
    @dashapp.callback(
        [
            Output('parameter-1', 'placeholder'),
            Output('parameter-2', 'placeholder'),
            Output('parameter-3', 'placeholder'),
            Output('parameter-4', 'placeholder'),
            Output('parameter-1', 'style'),
            Output('parameter-2', 'style'),
            Output('parameter-3', 'style'),
            Output('parameter-4', 'style')
        ],
        [Input('feature-sets', 'value'), Input('variable', 'value')],
        [State('league', 'value')]
    )
    def config_parameters(feature_set, variable, league):
        return ConfigCallbacks.parameters(feature_set, variable, league)

    # Populate with history
    @dashapp.callback(
        [
            Output('history-data', 'children'),
            Output('history-x', 'options'),
            Output('history-y', 'options')
        ],
        [Input('update-history-data', 'n_clicks'), Input('league', 'value')],
        [State('team', 'value'), State('opponent', 'value')]
    )
    def history_data(trigger, league, team, opponent):
        return DataCallbacks.history(league, team, opponent)

    # Make the figure
    @dashapp.callback(
        [
            Output('history-fig', 'figure'),
            Output('history-fig', 'style'),
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
        return PlotCallbacks.history(df, x, y)

    # Populate with results
    @dashapp.callback(
        [
            Output('results-win-data', 'children'),
            Output('results-margin-data', 'children'),
            Output('results-total-points-data', 'children')
        ],
        [Input('update-results-data', 'n_clicks')],
        [
            State('league', 'value'),
            State('feature-sets', 'value'),
            State('team', 'value'),
            State('opponent', 'value'),
            State('variable', 'value'),
            State('parameter-1', 'placeholder'),
            State('parameter-1', 'value'),
            State('parameter-2', 'placeholder'),
            State('parameter-2', 'value'),
            State('parameter-3', 'placeholder'),
            State('parameter-3', 'value'),
            State('parameter-4', 'placeholder'),
            State('parameter-4', 'value')
        ]
    )
    def results_data(trigger, league, feature_set, team, opponent, variable, *parameters):
        return DataCallbacks.results(league, feature_set, team, opponent, variable, *parameters)

    # Win figure
    @dashapp.callback(
        [Output('win-fig', 'figure'), Output('win-fig', 'style')],
        [Input('results-win-data', 'children')],
        [State('variable', 'value')]
    )
    def win_figure(df, variable):
        return PlotCallbacks.win_figure(df, variable)

    # margin figure
    @dashapp.callback(
        [Output('margin-fig', 'figure'), Output('margin-fig', 'style')],
        [Input('results-margin-data', 'children'), Input('win-fig', 'hoverData')]
    )
    def conditioned_margin_figure(df, variable_val):
        return PlotCallbacks.conditioned_margin_figure(df, variable_val)

    # Total Points figure
    @dashapp.callback(
        [Output('total-points-fig', 'figure'), Output('total-points-fig', 'style')],
        [Input('results-total-points-data', 'children'), Input('win-fig', 'hoverData')]
    )
    def total_points_figure(df, variable_val):
        return PlotCallbacks.total_points_figure(df, variable_val)

    return dashapp.server

