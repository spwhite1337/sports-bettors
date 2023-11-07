from sports_bettors.analytics.run import run, analysis
from config import logger


def refresh_college_data():
    # Section for refreshing College football data
    import re
    import cfbd
    import pandas as pd
    from tqdm import tqdm

    configuration = cfbd.Configuration()
    configuration.api_key['Authorization'] = None
    configuration.api_key_prefix['Authorization'] = 'Bearer'

    api_instance = cfbd.BettingApi(cfbd.ApiClient(configuration))
    years = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
    conferences = ['ACC', 'B12', 'B1G', 'SEC', 'Pac-10', 'PAC']
    season_type = 'regular'
    df = []
    for year in tqdm(years):
        for conference in tqdm(conferences):
            api_response = api_instance.get_lines(year=year, season_type=season_type, conference=conference)
            records = []
            for b in api_response:
                record = {
                    'gameday': b.start_date,
                    'game_id': str(year) + '_' + re.sub(' ', '', b.away_team) + '_' + re.sub(' ', '', b.home_team),
                    'away_conference': b.away_conference,
                    'away_team': b.away_team,
                    'away_score': b.away_score,
                    'home_conference': b.home_conference,
                    'home_team': b.home_team,
                    'home_score': b.home_score
                }
                for line in b.lines:
                    record['away_moneyline'] = line.away_moneyline
                    record['home_moneyline'] = line.home_moneyline
                    record['formatted_spread'] = line.formatted_spread
                    record['over_under'] = line.over_under
                    record['provider'] = line.provider
                    # The spreads have different conventions but we want them relative to the away team
                    spread = line.formatted_spread.split(' ')[-1]
                    if spread in ['-null', 'null']:
                        record['spread_line'] = None
                    else:
                        if b.away_team in line.formatted_spread:
                            record['spread_line'] = float(spread)
                        else:
                            record['spread_line'] = -1 * float(spread)
                    records.append(record.copy())
            df.append(pd.DataFrame.from_records(records))
    df = pd.concat(df).drop_duplicates().reset_index(drop=True)
    df['gameday'] = pd.to_datetime(df['gameday']).dt.date
    return df


def refresh():
    analysis()
    for league in ['nfl', 'college_football']:
        for model in ['spread', 'over']:
            logger.info(f'Creating Model for {league}, {model}')
            run(league, model, overwrite=True)


if __name__ == '__main__':
    refresh()
