# Sports Bettors

Project to create betting aids for my favorite sports. The online gambling sites I use typically implement options 
to (i) bet on a winner, (ii) bet against a spread or (iii) bet an over / under on total points. This project helps one
combine these bets by conditioning a probability of one on the results of the other. Additionally, one can formalize
intuitions about a team's performance by, for example, conditioning a win probability on 200 rushing yards for a 
favorite team.

As of 7/14/2020; models are available for two leagues: `college_football` and `nfl`. The code implements PyStan to 
fit Hierarchical Bayesian models with the team of interest serving as the random effect, and various combinations of 
game statistics as the fixed effects. The team could be also be specified by opponent to shift the meaning of 
conditions. For `college_football`, the rank of the team / opponent can also be used with "unranked" opponents 
comprising the largest group.

Each experiment assesses a random-effect, feature-set, and response combination. The results of which can be seen in an
automatically generated diagnostics report. A light-weight predictor object is also generated which will approximate the
posterior probabilities for a new input vector without full-sampling of the posterior (There is a reason Stan isn't 
often used in ML production).

<img src="/docs/sports_bettors_logo.jpg" alt="Sports Betting" width="512">

---
<p>
    <img src="docs/pystan_logo.jpg", alt="Pystan Logo" with="256">
</p>

---

## Procedure

- Python 3.5
- `cd sports-bettors`
- `pip install -e .`
- `aws configure` (enter AWS keys in prompt, email me for a pair)
- On Windows: `conda install libpython m2w64-toolchain -c msys2`
    - This installs a C++ compiler (This requires conda, see pystan docs for other distributions)
- Optional: Download datasets and results sets with `sb_download --aws`; `--windows` if on windows machine and 
`--dryrun` to test the download.
    - Skip datasets or results sets with `sb_download --aws --skipdata --skipresults`

## Get Data

- Download raw data with: `sb_download --league [league]`
    - College Football data is downloaded from https://api.collegefootballdata.com/api/docs/?url=/api-docs.json
    - NFL Data is scraped from https://www.pro-football-reference.com/. If the web front-end changes this download 
    script will need to be modified
- Curate data with `sb_curate --league [league]`

## Run Experiments

- Fit models with `sb_run_experiments --league [league]`
- Optional: overwrite previously fit models with `sb_run_experiments --league [league] --overwrite`
- Generate light-weight predictor objects with `sb_generate_predictors --league [league]`

## Unit Tests

- `cd tests`
- `python -m unittest`
- Unit tests generate plots of simulated posteriors vs. approximated predictions from the predictor objects. The 
two should be close. 
- They also outline example use cases of the predictor objects.

## Predictions

You can access the api from the command line by specifying (i) the league, (ii) the Random Effect type (e.g. `team` or
`opponent`), and (iii) the feature set of interest (e.g. `RushOnly`, `PointsScored`). You will then be prompted to input 
values for each of the features in the selected feature set. 
 - Note: RandomEffect values must be put in exactly. In the future I will add a parser but for the most part the nfl
 teams are three letters, all caps (CHI, GNB) and college teams are camelcase (Iowa, OhioState).

The result for each model will be outputs as an approximation of the posterior distribution of the response variable. 
Note that for `Win` response, the outputs are log-odds.

```
>sb_predict --league college_football --random_effect team --feature_set RushOnly --display_output
Input Value for RandomEffect (team): Iowa
Input Value for rushingYards: 150
Input Value for rushingAttempts: 30
INFO:config:Loading predictor set for college_football
INFO:config:Input: {'RandomEffect': 'Iowa', 'rushingYards': 150.0, 'rushingAttempts': 30.0}
INFO:config:Output:
{   ('team', 'RushOnly', 'LossMargin'): {   'mu': {   'lb': 5.203407115684713,
                                                      'mean': 6.693706556567932,
                                                      'ub': 8.184005997451152},
                                            'sigma': {   'lb': 12.933691310049893,
                                                         'mean': 13.018212804017745,
                                                         'ub': 13.102734297985597}},
    ('team', 'RushOnly', 'Margin'): {   'mu': {   'lb': 6.585329165536013,
                                                  'mean': 7.641162796263397,
                                                  'ub': 8.696996426990781},
                                        'sigma': {   'lb': 18.085038403813193,
                                                     'mean': 18.171557216983697,
                                                     'ub': 18.2580760301542}},
    ('team', 'RushOnly', 'TotalPoints'): {   'mu': {   'lb': 46.89018052371675,
                                                       'mean': 47.898364227578945,
                                                       'ub': 48.90654793144114},
                                             'sigma': {   'lb': 16.932600687085372,
                                                          'mean': 17.003502059890415,
                                                          'ub': 17.07440343269546}},
    ('team', 'RushOnly', 'Win'): {   'mu': {   'lb': 0.2877182200285193,
                                               'mean': 0.43064431853860385,
                                               'ub': 0.5735704170486884}},
    ('team', 'RushOnly', 'WinMargin'): {   'mu': {   'lb': 17.350635208395776,
                                                     'mean': 18.049121264808306,
                                                     'ub': 18.747607321220833},
                                           'sigma': {   'lb': 12.933624296832694,
                                                        'mean': 13.012824812353006,
                                                        'ub': 13.09202532787332}}}

>sb_predict --league nfl --random_effect team --feature_set RushOnly --display_output
Input Value for RandomEffect (team): CHI
Input Value for rushYards: 150
Input Value for rushAttempts: 30
INFO:config:Loading predictor set for nfl
INFO:config:Input: {'rushAttempts': 30.0, 'RandomEffect': 'CHI', 'rushYards': 150.0}
INFO:config:Output:
{   ('team', 'RushOnly', 'LossMargin'): {   'mu': {   'lb': 8.823715295603904,
                                                      'mean': 9.433845355415919,
                                                      'ub': 10.043975415227932},
                                            'sigma': {   'lb': 9.020737754734782,
                                                         'mean': 9.076146979627058,
                                                         'ub': 9.131556204519335}},
    ('team', 'RushOnly', 'Margin'): {   'mu': {   'lb': 1.5938844986531027,
                                                  'mean': 2.1024059929882357,
                                                  'ub': 2.6109274873233694},
                                        'sigma': {   'lb': 12.495524454026041,
                                                     'mean': 12.546819488200619,
                                                     'ub': 12.598114522375196}},
    ('team', 'RushOnly', 'TotalPoints'): {   'mu': {   'lb': 41.32055847012186,
                                                       'mean': 41.85741135654675,
                                                       'ub': 42.39426424297163},
                                             'sigma': {   'lb': 14.040654969318101,
                                                          'mean': 14.098706861493424,
                                                          'ub': 14.156758753668747}},
    ('team', 'RushOnly', 'Win'): {   'mu': {   'lb': 0.06459991900071245,
                                               'mean': 0.15732826127700533,
                                               'ub': 0.25005660355329823}},
    ('team', 'RushOnly', 'WinMargin'): {   'mu': {   'lb': 12.241533549717488,
                                                     'mean': 12.564914539722725,
                                                     'ub': 12.888295529727962},
                                           'sigma': {   'lb': 9.003596621370459,
                                                        'mean': 9.058188389559072,
                                                        'ub': 9.112780157747686}}}
```

## Dashboard

A dashboard is available to explore results. 

- `cd api`
- `set FLASK_APP=run.py` (`export` in bash)
- `flask run`
