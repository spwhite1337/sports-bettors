import pandas as pd
import plotly.express as px

from sports_bettors.dashboard.params import params, utils
from sports_bettors.dashboard.utils.history import populate as history_populate
from sports_bettors.dashboard.utils.results import ResultsPopulator

from config import Config


class ConfigCallbacks(object):
    """
    Configure the layout to create inputs and display html elements
    """
    @staticmethod
    def dropdowns(league: str):
        """
        Populate dropdowns
        """
        team_opts = params[Config.sb_version]['team-opts'][league]
        feature_set_opts = params[Config.sb_version]['feature-sets-opts'][league]
        return team_opts, utils['show'], team_opts, utils['show'], feature_set_opts, utils['show']

    @staticmethod
    def variables(league: str, feature_set: str):
        """
        Populate and show a drop down that selects the "variable" to span for results
        """
        if (league is None) or (feature_set is None):
            return [], utils['no_show']
        variable_opts = params[Config.sb_version]['variable-opts'][league][feature_set]
        return variable_opts, utils['show']

    @staticmethod
    def parameters(feature_set: str, variable: str, league: str):
        """
        Populate and show inputs for defining parameters of the input model
        """
        if (league is None) or (feature_set is None) or (variable is None):
            return [None] * 4 + [utils['no_show']] * 4
        # Get parameters except for variable
        parameters = [
            p['label'] for p in params[Config.sb_version]['variable-opts'][league][feature_set] if p['value'] != variable
        ]
        # Fill displays and Nones
        displays = [utils['show']] * len(parameters) + [utils['no_show']] * (4 - len(parameters))
        parameters += [None] * (4 - len(parameters))

        return parameters + displays


class DataCallbacks(object):
    """
    Generate the historical and results-based data to display in the dashboard
    """
    @staticmethod
    def history(league: str, team: str, opponent: str):
        """
        Load data for historical matchups
        """
        df, x_opts, y_opts = history_populate(league, team, opponent)
        return df.to_json(), x_opts, y_opts

    @staticmethod
    def results(league: str, feature_set: str, team: str, opponent: str, variable: str, *parameters):
        """
        Calculate probabilities
        """
        if not all([league, feature_set, team, opponent, variable]):
            return pd.DataFrame().to_json(), pd.DataFrame().to_json(), pd.DataFrame().to_json()

        # Drop nones in parameters
        parameters = [p for p in parameters if p]

        # Convert to dictionary
        def _parse_parameters(p):
            p = {k: v for k, v in zip(p[::2], p[1::2])} if len(p) > 1 else {}
            # Convert keys, values
            p = {utils['feature_maps'][Config.sb_version][league][k]: int(v) for k, v in p.items()}
            return p
        parameters = _parse_parameters(parameters)

        # Results
        populator = ResultsPopulator(
            league=league,
            feature_set=feature_set,
            team=team,
            opponent=opponent,
            variable=variable,
            parameters=parameters
        )

        # Win probabilities
        df_win = populator.win()
        df_margins = populator.margins()
        df_points = populator.total_points()

        return df_win.to_json(), df_margins.to_json(), df_points.to_json()


class PlotCallbacks(object):
    """
    Generate plotly figures from history and results
    """
    @staticmethod
    def history(df, x: str, y: str):
        """
        Plot historical data
        """
        df = pd.read_json(df, orient='records')
        if df.shape[0] == 0:
            return utils['empty_figure'], utils['no_show'], utils['no_show']
        x = df.columns[0] if not x else x
        y = df.columns[0] if not y else y
        fig = px.scatter(df, x=x, y=y, color='Winner')
        return fig, utils['show'], utils['show']

    @staticmethod
    def win_figure(df, variable: str):
        """
        Plot results
        """
        df = pd.read_json(df, orient='records')
        if df.shape[0] == 0:
            fig = utils['empty_figure']
        else:
            # Win Probability Figure
            df = df.sort_values(variable)
            fig = px.line(df, x=variable, y='Win', error_y='WinUB', error_y_minus='WinLB')
        return fig

    @staticmethod
    def conditioned_margin_figure(df, variable_val):
        """
        Plot conditioned results for margins
        """
        df = pd.read_json(df)
        if df.shape[0] == 0:
            fig = utils['empty_figure']
        else:
            if variable_val is None:
                return utils['empty_figure']
            # Win margin figure
            variable_val = variable_val['points'][0]['x']
            df = df[df['variable_val'] == variable_val].sort_values('Margin')
            fig = px.line(df, x='Margin', y='Probability', error_y='Probability_UB', error_y_minus='Probability_LB',
                          color='Result')
        return fig

    @staticmethod
    def total_points_figure(df, variable_val):
        """
        Total points figure
        """
        df = pd.read_json(df)
        if df.shape[0] == 0:
            fig = utils['empty_figure']
        else:
            if variable_val is None:
                return utils['empty_figure']
            # Total Points figure
            variable_val = variable_val['points'][0]['x']
            df = df[df['variable_val'] == variable_val].sort_values('TotalPoints')
            fig = px.line(df, x='TotalPoints', y='Probability', error_y='Probability_UB',
                          error_y_minus='Probability_LB')
        return fig
