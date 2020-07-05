# Iowa Football Stats

Get Data:
 - Use this API to get box scores: https://api.collegefootballdata.com/api/docs/?url=/api-docs.json#/
 - For Iowa, get extra historical data
 
Engineer Features
 - Get stat differentials (rush yards, pass yards, turnover-margin, penalties diff) to predict point differential
  
Make a global model; a hierarchical model for the teams in the big ten; and a hierarchical model for the matchups.
Each successive model is more specific, but will have lower and lower data. 

The app will display the historical data for all teams and Iowa. Then it will display model results for each level
of hierarchical modeling to explore the relationship between spread and differentials. 

Will ultimately server to entertain and improve betting strategies when you "know" the stat differentials. 