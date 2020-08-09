import pandas as pd
from scipy.special import expit
from sports_bettors.dashboard.params import params, utils

from sports_bettors.api import SportsPredictor

from config import Config


def win_probability(predictor, league: str, variable: str, feature_set: str, parameters: dict, team: str,
                    opponent: bool):
    """
    Return df of win probability with fields []
    """
    records = []
    for var in params[Config.sb_version]['variable-ranges'][league][variable]:
        # Add variable to parameters
        parameters[variable] = var

        # Add derived features
        feature_creators = utils['feature_creators'][Config.sb_version][league][feature_set]
        for created_feature, creator in feature_creators.items():
            parameters[created_feature] = creator(parameters)

        # Configure inputs
        input_set = {
            'random_effect': 'team' if not opponent else 'opponent',
            'feature_set': feature_set,
            'inputs': {'RandomEffect': team}
        }
        input_set['inputs'].update(parameters)

        # Predict
        output = predictor.predict(**input_set)[('team' if not opponent else 'opponent', feature_set, 'Win')]

        # Wrangle output
        record = {
            'RandomEffect': team,
            variable: var,
            'WinLB_opp' if opponent else 'WinLB_team': expit(output['mu']['lb']),
            'Win_opp' if opponent else 'Win_team': expit(output['mu']['mean']),
            'WinUB_opp' if opponent else 'WinUB_team': expit(output['mu']['ub'])
        }
        records.append(record)

    return pd.DataFrame().from_records(records)


def normalize_win_prob(team: pd.DataFrame, opp: pd.DataFrame, variable: str) -> pd.DataFrame:
    df = team.drop('RandomEffect', axis=1).merge(opp.drop('RandomEffect', axis=1), on=variable, how='inner')
    # When controlling for opponent, 1 - P(Win) is the probability of the opponent winning. We force the sum of
    # These P(Win | team) + (1 - P(Win | Opponent) to be 1.
    df['Win'] = df['Win_team'] / (df['Win_team'] + (1 - df['Win_opp']))
    return df


def populate(league: str, feature_set: str, team: str, opponent: str, variable: str, parameters: dict):
    """
    Generate a json friendly dataframe of sports-bettors outputs
    """
    if not all([league, feature_set, team, opponent, variable]):
        return pd.DataFrame().from_records([])

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

    assert league in ['college_football', 'nfl']
    # Load predictor
    predictor = SportsPredictor(league=league)
    predictor.load()

    # Calculate probabilities for the team winning
    team_win = win_probability(predictor, league, variable, feature_set, parameters, team, opponent=False)

    # Calculate probabilities for the opponent winning
    opponent_win = win_probability(predictor, league, variable, feature_set, parameters, opponent, opponent=True)

    # Normalize together
    df_win = normalize_win_prob(team_win, opponent_win, variable)

    return df_win

