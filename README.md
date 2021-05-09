# NBA-RankSVM
Final for CIS519: Applied Machine Learning at the University of Pennsylvania.

The code first computes all pairings of teams, (team1, team2), and treats
the label of each pair to be the difference in their actual standing.

Afterwards, the code uses 3 fold cross-validation to evaluate an sklearn SVM model's ability
to accurately rank the NBA teams, on a per-decade basis starting with 1980-1989, 
and ending with 2010-2018. The evaluation metric used is mean Average Precision (mAP).

# Licensing
Under no circumstance, may this code be reused for an assignment of any kind.
