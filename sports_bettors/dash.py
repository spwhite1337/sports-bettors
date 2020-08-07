import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px

from sports_bettors.dashboard.project_params import params
from sports_bettors.dashboard.populate import populate


def add_sb_dash(server, routes_pathname_prefix: str = '/api/dash/sportsbettors/'):
    dashapp = dash.Dash(
        __name__,
        routes_pathname_prefix=routes_pathname_prefix,
        server=server
    )

    df = populate(league='nfl', feature_set='PointsScored', team='CHI', opponent='GNB', output_type='probability')
    fig = px.scatter(df, x='total_points', y='Win')

    dashapp.layout = html.Div(children=[
        html.H1('Hi From Dash (sports bettors)'),
        dcc.Graph(id='example-fig', figure=fig)
    ])

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
