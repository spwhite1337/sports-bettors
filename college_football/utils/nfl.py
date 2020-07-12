from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


class DownloadNFLData(object):
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

    # URL to Format
    url_template = "https://www.pro-football-reference.com/teams/{team}/"

    # Iterate Player Data Frame for Each Year Specified

    def run(self):
        nfl_df = pd.DataFrame()

        for team in self.team_codes:
            url = self.url_template.format(team=team)  # get the url

            html = urlopen(url)

            soup = BeautifulSoup(html, 'html.parser')

            column_headers = [th.getText() for th in
                              soup.findAll('thead', limit=1)[0].findAll('tr', attrs={'class': ''})[0].findAll('th')]

            data_rows = soup.findAll('tbody', limit=1)[0].findAll('tr')[0:]

            team_data = [[td.getText() for td in data_rows[i].findAll(['th', 'td'])]
                         for i in range(len(data_rows))]

            # Turn yearly data into a DataFrame

            year_df = pd.DataFrame(team_data, columns=column_headers)

            year_df = year_df.loc[(year_df['Div. Finish'] != 'Overall Rank') & (year_df['Div. Finish'] != 'Div. Finish')]

            year_df = year_df.infer_objects().reset_index(drop=True)

            nfl_df = pd.concat([nfl_df, year_df], axis=0, ignore_index=True)

            return nfl_df