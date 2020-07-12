import argparse

from college_football.utils.college.download import DownloadCollegeFootballData
from college_football.utils.nfl.download import DownloadNFLData


def download_cli():
    parser = argparse.ArgumentParser(prog='Download Football Data')
    parser.add_argument('--league', default='college')
    parser.add_argument('--retry', action='store_true')
    args = parser.parse_args()
    download(league=args.league, retry=args.retry)


def download(league: str, retry: bool):
    if league == 'college':
        downloader = DownloadCollegeFootballData()
        if not retry:
            df_games = downloader.download_games()
            downloader.download_stats(df_games)
            downloader.download_rankings()
        else:
            downloader.retry_stats()
    elif league == 'nfl':
        downloader = DownloadNFLData()
        downloader.download_stats()
    else:
        raise NotImplementedError(league)
