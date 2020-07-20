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
>sb_predict --league nfl --random_effect team --feature_set RushOnly
Input Value for RandomEffect (team): CHI
Input Value for rushYards: 150
Input Value for rushAttempts: 30
INFO:config:Loading predictor set for nfl
INFO:config:{'rushYards': 150.0, 'RandomEffect': 'CHI', 'rushAttempts': 30.0}
INFO:config:{('team', 'RushOnly', 'LossMargin'): {'lb': -0.3078409089154306,
                                      'mean': 9.433845355415919,
                                      'ub': 19.175531619747268},
 ('team', 'RushOnly', 'Margin'): {'lb': -11.004230023722094,
                                  'mean': 2.1024059929882357,
                                  'ub': 15.209042009698566},
 ('team', 'RushOnly', 'TotalPoints'): {'lb': 27.163799716453113,
                                       'mean': 41.85741135654675,
                                       'ub': 56.55102299664038},
 ('team', 'RushOnly', 'Win'): {'lb': 0.06459991900071245,
                               'mean': 0.15732826127700533,
                               'ub': 0.25005660355329823},
 ('team', 'RushOnly', 'WinMargin'): {'lb': 3.004496014281422,
                                     'mean': 12.564914539722725,
                                     'ub': 22.12533306516403}}

>sb_predict --league college_football --random_effect team --feature_set RushOnly
Input Value for RandomEffect (team): Iowa
Input Value for rushingYards: 150
Input Value for rushingAttempts: 30
INFO:config:Loading predictor set for college_football
INFO:config:{'RandomEffect': 'Iowa', 'rushingYards': 150.0, 'rushingAttempts': 30.0}
INFO:config:{('team', 'RushOnly', 'LossMargin'): {'lb': -8.05546164046454,
                                      'mean': 6.693706556567932,
                                      'ub': 21.442874753600403},
 ('team', 'RushOnly', 'Margin'): {'lb': -12.014449857615572,
                                  'mean': 7.641162796263397,
                                  'ub': 27.296775450142366},
 ('team', 'RushOnly', 'TotalPoints'): {'lb': 29.50366178535086,
                                       'mean': 47.898364227578945,
                                       'ub': 66.29306666980703},
 ('team', 'RushOnly', 'Win'): {'lb': 0.23971163857666822,
                               'mean': 0.43064431853860385,
                               'ub': 0.6215769985005395},
 ('team', 'RushOnly', 'WinMargin'): {'lb': 3.6044349504541238,
                                     'mean': 18.049121264808306,
                                     'ub': 32.493807579162485
```
