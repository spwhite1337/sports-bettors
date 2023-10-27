from typing import Optional
from sports_bettors.analytics.bets.bets import Bets
from sports_bettors.analytics.eda.eda import Eda
from sports_bettors.analytics.model.policy import Policy
from sports_bettors.analytics.model import Model


def analysis():
    for league in ['nfl', 'college_football']:
        Eda(league=league).analyze()
    Bets().analyze()


def run(league: str = 'nfl', response: str = 'spread', run_shap: bool = False, overwrite: bool = False):
    api = Policy(league=league, response=response, overwrite=overwrite)
    df, df_val, df_all = api.fit_transform()
    api.train(df)
    api.validate(df, df_val, df_all, run_shap=run_shap)
    api.save_results()


def predict():
    Model().predict_next_week()


if __name__ == '__main__':
    run()
