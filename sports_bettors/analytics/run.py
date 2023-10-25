from typing import Optional
from sports_bettors.analytics.bets.bets import Bets
from sports_bettors.analytics.eda.eda import Eda
from sports_bettors.analytics.model.validate import Validate


def analysis():
    Eda().analyze()
    Bets().analyze()


def run(league: str = 'nfl', run_shap: bool = False, overwrite: bool = False):
    api = Validate(league=league, overwrite=overwrite)
    df, df_val, df_all = api.fit_transform()
    api.train(df)
    api.validate(df, df_val, df_all, run_shap=run_shap)
    api.save_results()


def predict(league: Optional[str] = None):
    leagues = ['nfl', 'college_football'] if league is None else [league]
    for league in leagues:
        api = Validate(league=league).load_results()
        api.predict_next_week()


if __name__ == '__main__':
    run()
