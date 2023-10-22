import os
import re
import argparse

from sports_bettors.utils.college_football.download import DownloadCollegeFootballData
from sports_bettors.utils.nfl.download import DownloadNFLData

from config import Config, logger


def download_cli():
    parser = argparse.ArgumentParser(prog='Download Football Data')
    parser.add_argument('--league', required=False)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--retry', action='store_true')
    parser.add_argument('--aws', action='store_true')
    parser.add_argument('--windows', action='store_true')
    parser.add_argument('--skipdata', action='store_true')
    parser.add_argument('--skipresults', action='store_true')
    parser.add_argument('--dryrun', action='store_true')
    args = parser.parse_args()

    if args.aws:
        # General commands
        sync_base = 'aws s3 sync '
        dryrun_arg = ' --dryrun'
        results_sync = '{} {}'.format(Config.CLOUD_RESULTS, Config.RESULTS_DIR)
        data_sync = '{} {}'.format(Config.CLOUD_DATA, Config.DATA_DIR)
        include_flags = " --exclude '*' --include 'sports_bettors/*'"

        if args.windows:
            include_flags = re.sub("'", "", include_flags)

        if not args.skipdata:
            logger.info('Downloading Data from AWS')
            sb_sync = sync_base + data_sync + include_flags
            sb_sync += dryrun_arg if args.dryrun else ''
            logger.info(sb_sync)
            os.system(sb_sync)
        if not args.skipresults:
            logger.info('Downloading Results from AWS')
            sb_sync = sync_base + results_sync + include_flags
            sb_sync += dryrun_arg if args.dryrun else ''
            logger.info(sb_sync)
            os.system(sb_sync)
    else:
        if args.league is None:
            ValueError('league argument required if not syncing with AWS')
        download(league=args.league, retry=args.retry, overwrite=args.overwrite)


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
