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

## Procedure

- Python 3.5
- `cd sports-bettors`
- `pip install -e .`
- On Windows: `conda install libpython m2w64-toolchain -c msys2`
    - This install a C++ compiler (This requires conda, see pystan docs for more info)

## Get Data

- Download data with: `sb_download --league [league]`
    - College Football data is downloaded from https://api.collegefootballdata.com/api/docs/?url=/api-docs.json
    - NFL Data is scraped from https://www.pro-football-reference.com/. If the web front-end changes this download 
    script will need to be modified
- Curate data with `sb_curate --league [league]`

## Run Experiments

- Fit models and generate predictor objects with `sb_run_experiments --league [league]`
- Optional args to facilitate batching the fit of all models
    - overwrite previously fit models with `sb_run_experiments --league [league] --overwrite`
    - Fit predictor objects for the models created so far with `sb_run_experiments --league [league] --skipfit`

## Unit Tests

- `cd tests`
- `python -m unittest`
- Unit tests generate plots of simulated posteriors vs. approximated predictions from the predictor objects. The 
two should be close. 
- They also outline example use cases of the predictor objects.

## Predictions

- TODO: 