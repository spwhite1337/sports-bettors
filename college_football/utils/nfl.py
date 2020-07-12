import os
import re
import pickle

from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
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
    dates = [datetime.strftime(d, '%Y%m%d') for d in pd.date_range(start='01-01-1985', end='01-01-2020', freq='1D')]

    # URL to Format
    base_url = "https://www.pro-football-reference.com/boxscores/{}0{}.htm"

    def download_stats(self):
        """
        Download raw data scraped from pro-football-reference
        """
        def uncomment_html(unparsed: str):
            return re.sub('<!--', '', re.sub('-->', '', unparsed))

        results, failed_urls, unparsed = [], [], []
        for team in tqdm(self.team_codes):
            for date in tqdm(self.dates):
                url = self.base_url.format(date, team)

                # Open url and parse
                try:
                    html = urlopen(url).read()
                    soup = BeautifulSoup(html)
                except Exception as err:
                    logger.info(err)
                    logger.info(url)
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

                    soup_box = BeautifulSoup(uncomment_html(str(all_team)))
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
                    results.append({date: result})
                except Exception as err:
                    logger.info(err)
                    logger.info(url)
                    unparsed.append(url)
                    continue

        logger.info('Saving {} Results.'.format(len(results)))
        with open(os.path.join(ROOT_DIR, 'data', 'nfl', 'raw.pkl'), 'rb') as fp:
            pickle.dump(results, fp)
