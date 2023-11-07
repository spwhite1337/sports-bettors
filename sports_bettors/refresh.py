from sports_bettors.analytics.run import run, analysis


def refresh():
    analysis()
    for league in ['nfl', 'college_football']:
        for model in ['spread', 'over']:
            run(league, model, overwrite=True)


if __name__ == '__main__':
    refresh()
