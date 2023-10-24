from sports_bettors.analytics.bets.bets import Bets
from sports_bettors.analytics.eda.eda import Eda
from sports_bettors.analytics.model.validate import Validate


def run():
    Eda().analyze()
    Bets().analyze()
    api = Validate()
    df, df_val, df_all = api.fit_transform()
    api.train(df)
    api.validate(df, df_val, df_all)
    api.save_results()


def predict():
    api = Validate().load_results()
    api.predict_next_week()


if __name__ == '__main__':
    run()
