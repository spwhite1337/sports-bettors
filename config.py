import os
import logging


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
    RAW_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
    CURATED_DIR = os.path.join(ROOT_DIR, 'data', 'curated')
    RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
    TEST_RESULTS_DIR = os.path.join(ROOT_DIR, 'tests', 'results')
    version = 'v1'
