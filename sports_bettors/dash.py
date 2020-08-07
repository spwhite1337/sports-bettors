import dash
import dash_core_components as dcc
import dash_html_components as html

from sports_bettors.dashboard.project_params import params
from sports_bettors.dashboard.populate import populate


def add_sb_dash(server, routes_pathname_prefix: str = '/api/dash/sportsbettors/'):
    dashapp = dash.Dash(
        __name__,
        routes_pathname_prefix=routes_pathname_prefix,
        server=server
    )
    dashapp.layout = html.Div(children=[
        html.H1('Hi From Dash (sports bettors)')
    ])

    df = populate(league='nfl', feature_set='PointsScored', random_effect='team',
                  random_effect_vals=['CHI', 'GNB'], output_type='probability'
                  )

    # On load; calculate default data set. The change in these parameters will be triggered by a button:
    #   League: college-football;
    #   Conditions: PointsScored;
    #   Teams: Iowa, Wisconsin, Michigan
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

    return dashapp.server
