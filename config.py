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
    def label_bet(league: str, response: str, p: float) -> Optional[str]:
        if response == 'spread':
            if league == 'college_football':
                if 3.5 <= p <= 7:
                    return 'Underdog'
                elif p <= -6:
                    return 'Favorite'
                else:
                    return 'No Bet'

            elif league == 'nfl':
                if p >= 2.5:
                    return 'Underdog'
                elif p <= -6 or (-1.5 >= p >= -2.5):
                    return 'Favorite'
                else:
                    return 'No Bet'
            else:
                return None

        elif response == 'over':
            return None
