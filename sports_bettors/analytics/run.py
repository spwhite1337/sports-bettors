from sports_bettors.analytics.bets.bets import Bets
from sports_bettors.analytics.eda.eda import Eda


def run():
    Eda().analyze()
    Bets().analyze()


if __name__ == '__main__':
    run()
