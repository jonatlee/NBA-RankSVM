from data_extractor import get_year_data
from ranksvm import RankSVM, run_3_fold_cv
import numpy as np

def run_experiment(clf):
    print('Converting CSV...')
    #   Try 1980-1989, 1990-1999, 2000-2009, and 2010-2018
    X1_west, X1_east, y1_west, y1_east = get_year_data(1980, 1989)
    X2_west, X2_east, y2_west, y2_east = get_year_data(1990, 1999)
    X3_west, X3_east, y3_west, y3_east = get_year_data(2000, 2009)
    X4_west, X4_east, y4_west, y4_east = get_year_data(2010, 2018)

    print('Running 5-fold CV...')
    
    print('1980-1989')
    mean_ave_precisions1_west = run_3_fold_cv(X1_west, y1_west, clf)
    mean_ave_precisions1_east = run_3_fold_cv(X1_east, y1_east, clf)
    
    print('1990-1999')
    mean_ave_precisions2_west = run_3_fold_cv(X2_west, y2_west, clf)
    mean_ave_precisions2_east = run_3_fold_cv(X2_east, y2_east, clf)
    
    print('2000-2009')
    mean_ave_precisions3_west = run_3_fold_cv(X3_west, y3_west, clf)
    mean_ave_precisions3_east = run_3_fold_cv(X3_east, y3_east, clf)
    
    print('2010-2018')
    mean_ave_precisions4_west = run_3_fold_cv(X4_west, y4_west, clf)
    mean_ave_precisions4_east = run_3_fold_cv(X4_east, y4_east, clf)
    
    return (mean_ave_precisions1_west, mean_ave_precisions1_east, 
        mean_ave_precisions2_west, mean_ave_precisions2_east, 
        mean_ave_precisions3_west, mean_ave_precisions3_east, 
        mean_ave_precisions4_west, mean_ave_precisions4_east)
        
# Get MAPs for each decade for each conference using all team stats
clf = RankSVM()
map1_west, map1_east, map2_west, map2_east, map3_west, map3_east, map4_west, map4_east = run_experiment(clf)

print('Average MAP for West 1980-1989:', np.mean(map1_west))
print('Average MAP for East 1980-1989:', np.mean(map1_east))

print('Average MAP for West 1990-1999:', np.mean(map2_west))
print('Average MAP for East 1990-1999:', np.mean(map2_east))

print('Average MAP for West 2000-2009:', np.mean(map3_west))
print('Average MAP for East 2000-2009:', np.mean(map3_east))

print('Average MAP for West 2010-2018:', np.mean(map4_west))
print('Average MAP for East 2010-2018:', np.mean(map4_east))