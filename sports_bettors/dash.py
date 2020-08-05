import dash
import dash_html_components as html


def add_sb_dash(server, routes_pathname_prefix: str = '/api/dash/sportsbettors/'):
    dashapp = dash.Dash(
        __name__,
        routes_pathname_prefix=routes_pathname_prefix,
        server=server
    )
    dashapp.layout = html.H1('Hi From Dash (sports bettors)')

    return dashapp.server
