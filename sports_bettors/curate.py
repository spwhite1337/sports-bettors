import argparse

from college_football.utils.college_football.curate import curate_college
from college_football.utils.nfl.curate import curate_nfl


def curate_data():
    """
    Curate downloaded data into a set for modeling / plotting.
    """
    parser = argparse.ArgumentParser(prog='Football Curated')
    parser.add_argument('--league', default='sports_bettors')
    args = parser.parse_args()

    if args.league == 'sports_bettors':
        curate_college()
    elif args.league == 'nfl':
        curate_nfl()
    else:
        raise NotImplementedError('No {}'.format(args.league))
