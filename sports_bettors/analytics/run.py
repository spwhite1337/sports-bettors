from sports_bettors.analytics.bets.bets import Bets
from sports_bettors.analytics.eda.eda import Eda
from sports_bettors.analytics.model.validate import Validate


def run():
    Eda().analyze()
    Bets().analyze()
    api = Validate()
    df, df_val, _ = api.fit_transform()
    api.train(df)
    api.validate()


if __name__ == '__main__':
    run()
