import os
import logging
from typing import Optional


# Setup logs
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Get root dir
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


class Config(object):
    ROOT_DIR = ROOT_DIR
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
    TEST_RESULTS_DIR = os.path.join(ROOT_DIR, 'tests', 'results')
    sb_version = 'v2'
    CLOUD_DATA = 's3://scott-p-white/website/data'
    CLOUD_RESULTS = 's3://scott-p-white/website/results'

    @staticmethod
    def label_bet(league: str, p: float) -> Optional[str]:
        if league == 'nfl':
            if p > 0.9:
                return 'home'
            elif p < -2.5:
                return 'away'
            else:
                return 'No Bet'
        elif league == 'college_football':
            if p > 0:
                return 'home'
            elif p < -0.5:
                return 'Amay'
            else:
                return 'No Bet'
        else:
            return None
