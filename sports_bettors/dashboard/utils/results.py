import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sports_bettors.dashboard.params import params, utils

from sports_bettors.api import SportsPredictor

from config import Config


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

    def _win_probabilities(self, is_opponent: bool) -> pd.DataFrame:
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
        df_team = self._win_probabilities(is_opponent=False)
        df_opp = self._win_probabilities(is_opponent=True)

        # Merge
        df = df_team.drop('RandomEffect', axis=1).merge(df_opp.drop('RandomEffect', axis=1),
                                                        on=self.variable, how='inner')

        # Normalize so that P(Win) + P(Lose) = 1.0 calculated from each perspective
        df['Win'] = df['Win_team'] / (df['Win_team'] + (1 - df['Win_opp']))
        df['WinUB'] = abs(df['WinUB_team'] / (df['WinUB_team'] + (1 - df['WinLB_opp'])) - df['Win'])
        df['WinLB'] = abs(df['Win'] - df['WinLB_team'] / (df['WinLB_team'] + (1 - df['WinUB_opp'])))

        return df

    def win_margin(self):
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
                'random_effect': 'team',
                'feature_set': self.feature_set,
                'inputs': {'RandomEffect': self.team}
            }
            input_set['inputs'].update(self.parameters)

            # Predict conditioned on Win
            output = self.predictor.predict(**input_set)[('team', self.feature_set, 'WinMargin')]
            mu, sigma = output['mu']['mean'], output['sigma']['mean']
            mu_lb = output['mu']['lb']
            mu_ub, sigma_ub = output['mu']['ub'], output['sigma']['ub']
            for win_margin in params[Config.sb_version]['response-ranges'][self.league]['WinMargin']:
                prob = 1 - norm.cdf(win_margin, mu, sigma)
                record = {
                    'variable_val': var,
                    'Margin': win_margin,
                    'Probability': prob,
                    'Probability_LB': prob - (1. - norm.cdf(win_margin, mu_lb, sigma_ub)),
                    'Probability_UB': (1. - norm.cdf(win_margin, mu_ub, sigma_ub)) - prob,
                    'Result': 'Win'
                }
                records.append(record)

            # Predict conditioned on Loss
            output = self.predictor.predict(**input_set)[('team', self.feature_set, 'LossMargin')]
            mu, sigma = output['mu']['mean'], output['sigma']['mean']
            mu_lb = output['mu']['lb']
            mu_ub, sigma_ub = output['mu']['ub'], output['sigma']['ub']
            for win_margin in params[Config.sb_version]['response-ranges'][self.league]['LossMargin']:
                prob = 1 - norm.cdf(win_margin, mu, sigma)
                record = {
                    'variable_val': var,
                    'Margin': win_margin,
                    'Probability': prob,
                    'Probability_LB': prob - (1. - norm.cdf(win_margin, mu_lb, sigma_ub)),
                    'Probability_UB': (1. - norm.cdf(win_margin, mu_ub, sigma_ub)) - prob,
                    'Result': 'Loss'
                }
                records.append(record)

        return pd.DataFrame().from_records(records)
