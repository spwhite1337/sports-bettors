import pandas as pd
import plotly.express as px

from sports_bettors.dashboard.utils.params import params, utils

from config import Config


class ConfigCallbacks(object):
    @staticmethod
    def dropdowns(league: str):
        team_opts = params[Config.version]['team-opts'][league]
        feature_set_opts = params[Config.version]['feature-sets-opts'][league]
        return team_opts, utils['show'], team_opts, utils['show'], feature_set_opts, utils['show']

    @staticmethod
    def variables(league, feature_set):
        if (league is None) or (feature_set is None):
            return [], utils['no_show']
        variable_opts = params[Config.version]['variable-opts'][league][feature_set]
        return variable_opts, utils['show']

    @staticmethod
    def parameters(feature_set, variable, league):
        if (league is None) or (feature_set is None) or (variable is None):
            return [None] * 4 + [None] * 4 + [utils['no_show']] * 4
        # Get parameters except for variable and fill with Nones
        parameters = [p['label'] for p in params[Config.version]['variable-opts'][league][feature_set]
                      if p['value'] != variable]
        # Fill displays and Nones
        displays = [utils['show']] * len(parameters) + [utils['no_show']] * (4 - len(parameters))
        parameters += [None] * (4 - len(parameters))

        return parameters + parameters + displays


class DataCallbacks(object):
    @staticmethod
    def history(df, x, y):
        df = pd.read_json(df, orient='records')
        if df.shape[0] == 0:
            return utils['empty_figure'], utils['no_show'], utils['no_show']
        x = df.columns[0] if not x else x
        y = df.columns[0] if not y else y
        fig = px.scatter(df, x=x, y=y, color='Winner')
        return fig, utils['show'], utils['show']

    @staticmethod
    def results():
        pass


class PlotCallbacks(object):
    pass
