import os
import logging

from sports_bettors.utils.college_football.models import CollegeFootballBettingAid
from sports_bettors.utils.nfl.models import NFLBettingAid

# Setup logs
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Get root dir
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

version = 'v1'

# Dictionary of compatible leagues
betting_aids = {'nfl': NFLBettingAid, 'college_football': CollegeFootballBettingAid}


class Config(object):
    pass
