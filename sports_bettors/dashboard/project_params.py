from sports_bettors.utils.college_football.models import CollegeFootballBettingAid
from sports_bettors.utils.nfl.models import NFLBettingAid


params = {
    'college_football': {
        'label': 'College Football',
        'conditions': CollegeFootballBettingAid.feature_sets.values(),
        'random_effects': CollegeFootballBettingAid.random_effects,
        'random_effect_vals': {
            'Iowa': 'Iowa',
            'Wisconsin': 'Wisconsin',
            'Michigan': 'Michigan'
        }
    },
    'nfl': {
        'label': 'NFL',
        'conditions': NFLBettingAid.feature_sets.values(),
        'random_effects': NFLBettingAid.random_effects,
        'random_effect_vals': {
            'Chicago Bears': 'CHI',
            'Green Bay Packers': 'GNB'
        },
        'responses': NFLBettingAid.responses
    }
}
