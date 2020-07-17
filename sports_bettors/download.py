import os
import argparse

from sports_bettors.utils.college_football.download import DownloadCollegeFootballData
from sports_bettors.utils.nfl.download import DownloadNFLData

from config import Config, logger


def download_cli():
    parser = argparse.ArgumentParser(prog='Download Football Data')
    parser.add_argument('--league', required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--retry', action='store_true')
    parser.add_argument('--aws', action='store_true')
    args = parser.parse_args()

    if args.aws:
        logger.info('Downloading Data from AWS')
        include_flags = '--include college_football/* --include nfl/*'
        aws_sync = 'aws s3 {} {} {}'.format(Config.CLOUD_DATA, Config.DATA_DIR, include_flags)
        os.system(aws_sync)
        logger.info('Downloading Results from AWS')
        include_flags = '--include aid_{}.pkl'.format(Config.version)
        aws_sync = 'aws s3 {} {} {}'.format(Config.CLOUD_RESULTS, Config.RESULTS_DIR, include_flags)
        os.system(aws_sync)
    else:
        download(league=args.league, retry=args.retry)


def download(league: str, retry: bool, overwrite: bool = False):
    if league == 'college_football':
        downloader = DownloadCollegeFootballData()
        if not retry:
            df_games = downloader.download_games()
            downloader.download_stats(df_games)
            downloader.download_rankings()
        else:
            downloader.retry_stats()
    elif league == 'nfl':
        downloader = DownloadNFLData(overwrite=overwrite)
        downloader.download_stats()
    else:
        raise NotImplementedError(league)
