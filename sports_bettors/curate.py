import argparse

from sports_bettors.utils.college_football.curate import curate_college
from sports_bettors.utils.nfl.curate import curate_nfl


def curate_data():
    """
    Curate downloaded data into a set for modeling / plotting.
    """
    parser = argparse.ArgumentParser(prog='Football Curated')
    parser.add_argument('--league', required=True)
    args = parser.parse_args()

    if args.league == 'college_football':
        curate_college()
    elif args.league == 'nfl':
        curate_nfl()
    else:
        raise NotImplementedError('No {}'.format(args.league))
