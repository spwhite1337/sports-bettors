import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sports_bettors.dashboard.params import params, utils

from sports_bettors.api import SportsPredictor

from config import Config


class ResultsPopulator(object):
    response_ranges = params[Config.sb_version]['response-ranges']

    def __init__(self,
                 league: str,
                 feature_set: str,
                 team: str,
                 opponent: str,
                 variable: str,
                 parameters: dict
                 ):
        assert league in ['nfl', 'college_football']
        self.league = league
        self.feature_set = feature_set
        self.feature_creators = utils['feature_creators'][Config.sb_version][self.league][self.feature_set]
        self.team = team
        self.opponent = opponent
        self.variable = variable
        self.variable_vals = params[Config.sb_version]['variable-ranges'][self.league][self.variable]
        self.parameters = parameters
        self.predictor = SportsPredictor(league=league, version=Config.sb_version)
        self.predictor.load()

    def _derived_features(self):
        """
        Add derived features to parameters
        """
        for created_feature, creator in self.feature_creators.items():
            self.parameters[created_feature] = creator(self.parameters)

    def _win(self, is_opponent: bool) -> pd.DataFrame:
        """
        Calculate win probabilities for a team or an opponent
        """
        records = []
        for var in self.variable_vals:
            # Add variable to parameters
            self.parameters[self.variable] = var
            # Add derived features
            self._derived_features()
            # Configure inputs
            input_set = {
                'random_effect': 'team' if not is_opponent else 'opponent',
                'feature_set': self.feature_set,
                'inputs': {'RandomEffect': self.team if not is_opponent else self.opponent}
            }
            input_set['inputs'].update(self.parameters)
            # Predict
            output = self.predictor.predict(**input_set)[
                ('team' if not is_opponent else 'opponent', self.feature_set, 'Win')
            ]
            # Wrangle output
            record = {
                'RandomEffect': self.team if is_opponent else self.opponent,
                self.variable: var,
                'WinLB_opp' if is_opponent else 'WinLB_team': expit(output['mu']['lb']),
                'Win_opp' if is_opponent else 'Win_team': expit(output['mu']['mean']),
                'WinUB_opp' if is_opponent else 'WinUB_team': expit(output['mu']['ub'])
            }
            records.append(record)

        return pd.DataFrame().from_records(records)

    def win(self) -> pd.DataFrame:
        """
        Calculate win probability as a function of `variable`
        """
        if not all([self.league, self.feature_set, self.team, self.opponent, self.variable]):
            return pd.DataFrame().from_records([])

        # Get win probability from the team and opponent's perspectives
        df_team = self._win(is_opponent=False)
        df_opp = self._win(is_opponent=True)

        # Merge
        df = df_team.drop('RandomEffect', axis=1).merge(df_opp.drop('RandomEffect', axis=1),
                                                        on=self.variable, how='inner')

        # Normalize so that P(Win) + P(Lose) = 1.0 calculated from each perspective
        df['Win'] = df['Win_team'] / (df['Win_team'] + (1 - df['Win_opp']))
        df['WinUB'] = abs(df['WinUB_team'] / (df['WinUB_team'] + (1 - df['WinLB_opp'])) - df['Win'])
        df['WinLB'] = abs(df['Win'] - df['WinLB_team'] / (df['WinLB_team'] + (1 - df['WinUB_opp'])))

        return df

    def _margins(self, is_opponent: bool) -> pd.DataFrame:
        """
        Probability of win margins
        """
        records = []
        for var in self.variable_vals:
            # Add variable to parameters
            self.parameters[self.variable] = var
            # Add derived features
            self._derived_features()
            # Configure inputs
            input_set = {
                'random_effect': 'team' if not is_opponent else 'opponent',
                'feature_set': self.feature_set,
                'inputs': {'RandomEffect': self.team if not is_opponent else self.opponent}
            }
            input_set['inputs'].update(self.parameters)

            # Probabilities for each margin-type
            for margin_type in ['WinMargin', 'LossMargin', 'Margin']:
                output = self.predictor.predict(**input_set)[(
                    'team' if not is_opponent else 'opponent',
                    self.feature_set,
                    margin_type
                )]
                mu, sigma = output['mu']['mean'], output['sigma']['mean']
                mu_lb = output['mu']['lb']
                mu_ub, sigma_ub = output['mu']['ub'], output['sigma']['ub']
                for margin in params[Config.sb_version]['response-ranges'][self.league][margin_type]:
                    prob = 1 - norm.cdf(margin, mu, sigma)
                    record = {
                        'variable_val': var,
                        'Margin': margin if margin_type != 'LossMargin' else -margin,
                        'Probability': prob,
                        'Probability_LB': prob - (1. - norm.cdf(margin, mu_lb, sigma_ub)),
                        'Probability_UB': (1. - norm.cdf(margin, mu_ub, sigma_ub)) - prob,
                        'Result': {'WinMargin': 'Win', 'LossMargin': 'Loss', 'Margin': 'Any'}.get(margin_type)
                    }
                    records.append(record)

        return pd.DataFrame().from_records(records)

    def margins(self) -> pd.DataFrame:
        """
        Calculate probability of certain margins conditioned and not-conditioned on winners.
        """
        df_team = self._margins(is_opponent=False)
        df_opp = self._margins(is_opponent=True)
        df = pd.concat([df_team, df_opp])
        df = df.groupby(['variable_val', 'Margin', 'Result']).agg(
            Probability=('Probability', 'mean'),
            Probability_LB=('Probability_LB', 'mean'),
            Probability_UB=('Probability_UB', 'mean')
        ).reset_index()

        return df

    def _total_points(self, is_opponent: bool) -> pd.DataFrame:
        """
        Total points
        """
        if self.feature_set == 'PointsScored':
            return pd.DataFrame().from_records([])

        records = []
        for var in self.variable_vals:
            # Add variable to parameters
            self.parameters[self.variable] = var
            # Add derived features
            self._derived_features()
            # Configure inputs
            input_set = {
                'random_effect': 'team' if not is_opponent else 'opponent',
                'feature_set': self.feature_set,
                'inputs': {'RandomEffect': self.team if not is_opponent else self.opponent}
            }
            input_set['inputs'].update(self.parameters)
            # Results
            output = self.predictor.predict(**input_set)[(
                'team' if not is_opponent else 'opponent',
                self.feature_set,
                'TotalPoints'
            )]
            mu, sigma = output['mu']['mean'], output['sigma']['mean']
            mu_lb = output['mu']['lb']
            mu_ub, sigma_ub = output['mu']['ub'], output['sigma']['ub']
            for total_points in params[Config.sb_version]['response-ranges'][self.league]['TotalPoints']:
                prob = 1 - norm.cdf(total_points, mu, sigma)
                record = {
                    'variable_val': var,
                    'TotalPoints': total_points,
                    'Probability': prob,
                    'Probability_LB': prob - (1. - norm.cdf(total_points, mu_lb, sigma_ub)),
                    'Probability_UB': (1. - norm.cdf(total_points, mu_ub, sigma_ub)) - prob,
                }
                records.append(record)

        return pd.DataFrame().from_records(records)

    def total_points(self) -> pd.DataFrame:
        """
        Total Points
        """
        df_team = self._total_points(is_opponent=False)
        df_opp = self._total_points(is_opponent=True)
        df = pd.concat([df_team, df_opp])
        df = df.groupby(['variable_val', 'TotalPoints']).agg(
            Probability=('Probability', 'mean'),
            Probability_LB=('Probability_LB', 'mean'),
            Probability_UB=('Probability_UB', 'mean')
        ).reset_index()

        return df
