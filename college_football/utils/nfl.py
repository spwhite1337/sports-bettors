import os
import re
import pickle

from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd

from tqdm import tqdm
from datetime import datetime

from config import ROOT_DIR, logger


class DownloadNFLData(object):
    # Teams to download
    team_codes = ['atl', 'buf', 'car',
                  'chi', 'cin', 'cle',
                  'clt', 'crd', 'dal',
                  'den', 'det', 'gnb',
                  'htx', 'jax', 'kan',
                  'mia', 'min', 'nor',
                  'nwe', 'nyg', 'nyj',
                  'oti', 'phi', 'pit',
                  'rai', 'ram', 'rav',
                  'sdg', 'sea', 'sfo',
                  'tam', 'was']

    # Dates to download
    dates = [datetime.strftime(d, '%Y%m%d') for d in pd.date_range(start='01-01-1985', end='01-01-2020', freq='1D')
             if d.month in [9, 10, 11, 12, 1, 2]]

    # URL to Format
    base_url = "https://www.pro-football-reference.com/boxscores/{}0{}.htm"

    def download_stats(self):
        """
        Download raw data scraped from pro-football-reference
        """
        def uncomment_html(content: str):
            return re.sub('<!--', '', re.sub('-->', '', content))

        results, failed_urls = {}, []
        for team in tqdm(self.team_codes):
            results_team, unparsed = {}, []
            for date in tqdm(self.dates):
                url = self.base_url.format(date, team)

                # Open url and parse
                try:
                    html = urlopen(url).read()
                    soup = BeautifulSoup(html, features='lxml')
                except Exception as err:
                    failed_urls.append(url)
                    continue

                # Parse output
                try:
                    # Get scores by quarter
                    score_tables = soup.findAll('table')[0]
                    quarter_headers = [th.text for th in score_tables.findAll('thead')[0].findAll('th') if len(th.text) > 0]
                    tbody = score_tables.findAll('tbody')[0]
                    quarter_values = [[cell.text for i, cell in enumerate(row.findAll('td'))]
                                      for row in tbody.findAll('tr')]

                    # Get box score
                    all_team = soup.find(id='all_team_stats')

                    soup_box = BeautifulSoup(uncomment_html(str(all_team)), features='lxml')
                    box = soup_box.findAll("table")[0]

                    # Get teams in order of [Away, Home]
                    teams = [th.text for th in box.findAll('thead')[0].findAll('th') if len(th.text) > 0]

                    # Get box score values
                    tbody = box.findAll('tbody')[0]
                    features = [header.text for header in tbody.findAll('th')]
                    values = [[cell.text for i, cell in enumerate(row.findAll('td'))] for row in tbody.findAll('tr')]

                    # Gather
                    result = {
                        'quarter_headers': quarter_headers,
                        'quarter_values': quarter_values,
                        'teams': teams,
                        'features': features,
                        'values': values
                    }
                    results_team[date] = result
                except Exception as err:
                    logger.info(err)
                    logger.info(url)
                    unparsed.append(url)
                    continue

            # Log each iteration
            results_dates = [pd.Timestamp(d) for d in results_team.keys()]
            logger.info('{}: {} First Game'.format(team, min(results_dates)))
            logger.info('{}: {} Games Returned'.format(team, len(results_dates)))
            logger.info('{}: {} Unparsed urls'.format(team, len(unparsed)))
            results[team] = results_team

        with open(os.path.join(ROOT_DIR, 'data', 'nfl', 'raw.pkl'), 'rb') as fp:
            pickle.dump(results, fp)
