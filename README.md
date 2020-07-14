# Sports Bettors

Project to create betting aids for my favorite sports. The online gambling sites I use typically implement options 
to (i) bet on a winner, (ii) bet against a spread or (iii) bet an over / under on total points. This project helps one
combine these bets by conditioning a probability of one on the results of the other. Additionally, one can formalize
intuitions about a team's performance by, for example, conditioning a win probability on 200 rushing yards for a 
favorite team.

As of 7/14/2020; models are available for two leagues: `college_football` and `nfl`

# Procedure

- Python 3.5
- `cd sports-bettors`
- `pip install -e .`
- On Windows: `conda install libpython m2w64-toolchain -c msys2`
    - This install a C++ compiler (This requires conda, see pystan docs for more info)

# Get Data

- Download data with: `sb_download --league [league]`
    - College Football data is downloaded from https://api.collegefootballdata.com/api/docs/?url=/api-docs.json
    - NFL Data is scraped from https://www.pro-football-reference.com/. If the web front-end changes this download 
    script will need to be modified
- Curate data with `sb_curate --league [league]`

# Run Experiments

- Fit models and generate predictor objects with `sb_run_experiments --league [league]`
- Optional: overwrite previously fit models with `sb_run_experiments --league [league] --overwrite`

# Unit Tests

- `cd tests`
- `python -m unittest`
- Unit tests generate plots of simulated posteriors vs. approximated predictions from the predictor objects. The 
two should be close. 

# Predictions

- TODO: 