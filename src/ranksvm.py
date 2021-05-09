from sklearn import svm
from itertools import permutations, combinations
import numpy as np
from operator import itemgetter


# Get all permutation pairs out of an array
def get_pairs(arr):
    return permutations(arr, 2)

# Transform data to pairs, where label of (x1, x2) is rank(x1) - rank(x2)
def data_to_pairs(X, y):
    X_pairs = [] 
    y_pairs = []
    
    pairs = get_pairs(np.arange(len(X)))
    
    for _, (index1, index2) in enumerate(pairs):
        name1 = X[index1][0]
        name2 = X[index2][0]
        X_pairs.append((X[index1][1:] + X[index2][1:]))
        result = y[name1] - y[name2]
        y_pairs.append(result)
            
    return X_pairs, y_pairs

# Transform just X data into pairs
def get_X_dict(X):
    X_dict_of_pairs = {}
    pairs = get_pairs(np.arange(len(X)))
    
    for _, (index1, index2) in enumerate(pairs):
        X_dict_of_pairs[(X[index1][0], X[index2][0])] = (X[index1][1:] + X[index2][1:])
        
    return X_dict_of_pairs

# Pairwise ranking SVM
class RankSVM(svm.LinearSVC):
    # Fit training data based off pairwise comparisons
    def fit(self, X, y):
        X_new, y_new = [], []
        for i in range(len(X)):
            X_pairs, y_pairs = data_to_pairs(X[i], y[i])
            X_new += X_pairs
            y_new += y_pairs
        super(RankSVM, self).fit(X_new, y_new)
        return self
    
    # Predict based off pairwise comparisons
    def predict(self, X):
        # Get all team names
        team_names = [X[i][0] for i in range(len(X))]
        
        # Setup dictionary of teams to 'score'
        dict_of_teams = {team: 0 for team in team_names}
        X_dict = get_X_dict(X)
        
        # Get relative rankings based off comparisons
        for (team1, team2) in X_dict.keys():
            ls_in = []
            ls_in.append(X_dict[(team1, team2)])
            dict_of_teams[team1] += super(RankSVM, self).predict(ls_in)
                
        # Determine the ranking of each team
        rankings = {}
        
        curr_rank = 1
        for team, _ in sorted(dict_of_teams.items(), key=itemgetter(1)):
            rankings[team] = curr_rank
            curr_rank += 1
        
        # Line up predictions with actuals
        predictions = [rankings[team] for team in team_names]
        
        return predictions

# Get the element missing from the subset of 3
def get_test_index(arr_3_indices, arr_2_indices):
    l = list(set(arr_3_indices) - set(arr_2_indices))
    return l[0]

# Calculate mean average precision of a ranking prediction
def mean_average_precision(actual, predicted):
    # Initialize list of average precisions
    average_precisions = []
    
    # Calculate all average precisions
    for i in range(1, len(actual) + 1):
        
        # Make actual and predicted lists of size i
        actual_i = actual[:i]
        predicted_i = predicted[:i]
        
        # Initialize score variables
        relevant_count = 0.0
        score = 0.0

        # Calculate an average precisoin
        for i, p in enumerate(predicted_i):
            if p in actual_i[:i + 1]:
                relevant_count += 1.0
                score += relevant_count/(i + 1.0)
            
        average_precisions.append(score / len(actual_i))
    
    return np.mean(average_precisions)

# Helper method to the helper method
def run_3_fold_cv_single_fold(X_train, y_train, clf, X_test, y_test):
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    actual = [value for value in y_test.values()]
    _map = mean_average_precision(actual, predicted)
    return _map

# Helper method
def run_3_fold_cv_helper(X_subset, y_subset, clf):
    # Initialize MAP array
    _maps = []
    
    # Initialize combinations of size 2
    combinations_of_size_2 = combinations(np.arange(3), 2)
    
    # Loop over size 2 combinations and actually call the method that gets the accuracies for that combination
    for combination in combinations_of_size_2:
        X_subset_of_2 = []
        y_subset_of_2 = []
        
        for _, j in enumerate(combination):
            X_subset_of_2.append(X_subset[j])
            y_subset_of_2.append(y_subset[j])
            
        test_index = get_test_index([0, 1, 2], list(combination))
        X_test = X_subset[test_index]
        y_test = y_subset[test_index]
        
        _map = run_3_fold_cv_single_fold(X_subset_of_2, y_subset_of_2, clf, X_test, y_test)
        _maps.append(_map)
        
    return np.mean(_maps)

# Run 3 CFV using MAP as the evaluation metric
def run_3_fold_cv(X, y, clf):
    # Initialize array arr = [<keys>]
    arr = X.keys()
    
    # Initialize array combinations to be all 3-sized combinations of years
    combinations_of_size_3 = combinations(arr, 3)
    
    # Initialize mean_average_precision array
    _maps = []
    
    # Loop over size 3 combinations and run 3 fold cv
    for combination in combinations_of_size_3:
        X_subset = []
        y_subset = []
        for _, j in enumerate(combination):
            X_subset.append(X[j])
            y_subset.append(y[j])
        _map = run_3_fold_cv_helper(X_subset, y_subset, clf)
        _maps.append(_map)
            
    return _maps