# Iowa Football Stats

Get Data:
 - Use this API to get box scores: https://api.collegefootballdata.com/api/docs/?url=/api-docs.json#/

Goals:
 - Create a betting aid that allows a user to input a team and conditioned stat differential:
    - Options include
        - One Team vs. anybody
            - This is fit as a hierarchical model with the team as a random effect
        - One Team vs. a ranked opponent
            - This is fit as a hierarchical model with the team and rank as a random effect
        - One Team vs. a Specific Opponent 
            - This is fit as a hierarchical model with the matchup as the random effect (will be tough)
        - If there are less than X-games then return an error because the model is not robust.
            
        - Stats Differentials
            - Can be one or many inputs like +100 rush yards, -50 Pass yards, +0 Turnover
            - If they only input some, can either force other inputs to 0; or can fit a model that doesn't use that stat.
        - Output the point differential
        - Also output a scatter plot in plotly/dashly that displays point differential vs. selected covariates with 
        optional filters. 
        
Necessary data:
    - One Team vs. Anybody:
        - Team DataSet Fields
            - Team
            - TeamScore
            - Opponent Score
            - HomeAdvantage
            - Team Stats
            - Opponent Stats
            - Engineered Features (differentials)
        - TeamRank DataSet Fields
            - Team
            - Opponent Rank
            - TeamScore
            - Opponent Score
            - HomeAdvantage
            - TeamStats
            - OpponentStats
            - Engineered Features (differentials)
        - Matchup DataSet Fields
            - Matchup
            - PointDifferential
            - StatDifferential
