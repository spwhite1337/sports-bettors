import argparse

from college_football.utils.college_football.download import DownloadCollegeFootballData
from college_football.utils.nfl.download import DownloadNFLData


def download_cli():
    parser = argparse.ArgumentParser(prog='Download Football Data')
    parser.add_argument('--league', default='college_football')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--retry', action='store_true')
    args = parser.parse_args()
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
