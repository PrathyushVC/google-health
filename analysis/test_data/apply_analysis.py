from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import sklearn.metrics
import roc
import pandas as pd
from statistics import mean, stdev, mode

def summary_stats(labels,roc_set_1,roc_set_2):
    """
    Compute summary statistics using DeLong test for multiple columns of ROC scores.

    Parameters:
    - labels (array-like): True labels used in the DeLong test.
    - roc_set_1 (DataFrame or array-like): First set of ROC (Receiver Operating Characteristic) scores.
    - roc_set_2 (DataFrame or array-like): Second set of ROC scores.

    Returns:
    - effects (list): List of effect sizes for each column.
    - cis (list): List of confidence intervals for each column.
    - p_values (list): List of p-values for each column.
    """
    effects,cis,p_values=[],[],[]
    for i in range(2,len(roc_set_1)):
        delong_test_results=roc.delong_test(y_true=labels,y_score_1=roc_set_1.iloc[:,i].to_numpy(),y_score_2=roc_set_2.iloc[:,i].to_numpy())
        effects.append(delong_test_results.effect)
        cis.append(delong_test_results.ci)
        p_values.append(delong_test_results.pvalue)
    return effects,cis,p_values

###########################################################
#run delong test on gargi
inf_validation=pd.read_excel('CCF_p_value_input_TIremission_random_seed_11.xlsx',header=0)
print(inf_validation.head())
inf_labels=inf_validation.iloc[:,0].to_numpy()
inf_radiomics=inf_validation.iloc[:,1].to_numpy()
inf_augmented=inf_validation.iloc[:,2].to_numpy()
inf_clinical=inf_validation.iloc[:,3].to_numpy()
inf_dt_rad_vs_clinical=roc.delong_test(y_true=inf_labels,y_score_1=inf_radiomics,y_score_2=inf_clinical)
inf_dt_augrad_vs_clinical=roc.delong_test(y_true=inf_labels,y_score_1=inf_augmented,y_score_2=inf_clinical)
inf_dt_augrad_vs_rad=roc.delong_test(y_true=inf_labels,y_score_1=inf_augmented,y_score_2=inf_radiomics)
unpack_1,unpack_2,unpack_3,unpack_4=inf_dt_rad_vs_clinical[0],inf_dt_rad_vs_clinical[1],inf_dt_rad_vs_clinical[2],inf_dt_rad_vs_clinical[3]

print(f"Inflammation DT Rad vs clinical{inf_dt_rad_vs_clinical},\nInflammation DT AUGRad vs clinical{inf_dt_augrad_vs_clinical},\nInflammation DT AUGRad vs Rad{inf_dt_augrad_vs_rad}")

#run delong test on the inflammation
'''

inf_validation=pd.read_csv('inf_validation.csv',header=None)
#print(inf_validation.head())
inf_labels=inf_validation.iloc[:,0].to_numpy()
inf_radiomics=inf_validation.iloc[:,1].to_numpy()
inf_augmented=inf_validation.iloc[:,2].to_numpy()
inf_radiologist=inf_validation.iloc[:,3].to_numpy()
#print(inf_labels)
#print(inf_radiologist)
inf_dt_rad_vs_radiologist=roc.delong_test(y_true=inf_labels,y_score_1=inf_radiomics,y_score_2=inf_radiologist)
inf_dt_augrad_vs_radiologist=roc.delong_test(y_true=inf_labels,y_score_1=inf_augmented,y_score_2=inf_radiologist)
inf_dt_augrad_vs_rad=roc.delong_test(y_true=inf_labels,y_score_1=inf_augmented,y_score_2=inf_radiomics)
unpack_1,unpack_2,unpack_3,unpack_4=inf_dt_rad_vs_radiologist[0],inf_dt_rad_vs_radiologist[1],inf_dt_rad_vs_radiologist[2],inf_dt_rad_vs_radiologist[3]

print(f"Inflammation DT Rad vs Radiologist{inf_dt_rad_vs_radiologist},\nInflammation DT AUGRad vs Radiologist{inf_dt_augrad_vs_radiologist},\nInflammation DT AUGRad vs Rad{inf_dt_augrad_vs_rad}")

###########################################################
#run delong test on inf in training
inf_training_radiomics=pd.read_csv('inf_training_radiomics.csv',header=None)
inf_labels=inf_training_radiomics.iloc[:,0].to_numpy()
inf_radiologist=inf_training_radiomics.iloc[:,1].to_numpy()
inf_training_aug_rad=pd.read_csv('inf_training_augrad.csv',header=None)

effects,cis,p_values=summary_stats(inf_labels,inf_training_radiomics.iloc[:,2:],inf_training_aug_rad.iloc[:,2:])

# Number of columns to duplicate
x = 100

# Duplicate the columns to create a matrix
duplicated_matrix = np.tile(inf_radiologist, (1, x)).reshape(-1, x)
effects_2,cis_2,p_values_2=summary_stats(inf_labels,inf_training_radiomics.iloc[:,2:],pd.DataFrame(duplicated_matrix))
effects_3,cis_3,p_values_3=summary_stats(inf_labels,inf_training_aug_rad.iloc[:,2:],pd.DataFrame(duplicated_matrix))
print(f"TRAINING Inflammation DT Rad vs Radiologist{(mean(effects_2),mean(p_values_2))},Inflammation DT AUGRad vs Radiologist{(mean(effects_3),mean(p_values_3))},Inflammation DT AUGRad vs Rad{(mean(effects),mean(p_values))}")







###########################################################
#run delong test on the fibrosis

fib_validation=pd.read_csv('fib_validation.csv',header=None)
#print(inf_validation.head())
fib_labels=fib_validation.iloc[:,0].to_numpy()
fib_radiomics=fib_validation.iloc[:,1].to_numpy()
fib_augmented=fib_validation.iloc[:,2].to_numpy()
fib_radiologist=fib_validation.iloc[:,3].to_numpy()
#print(fib_labels)
#print(fib_radiologist)

fib_dt_rad_vs_radiologist=roc.delong_test(y_true=fib_labels,y_score_1=fib_radiomics,y_score_2=fib_radiologist)
fib_dt_augrad_vs_radiologist=roc.delong_test(y_true=fib_labels,y_score_1=fib_augmented,y_score_2=fib_radiologist)
fib_dt_augrad_vs_rad=roc.delong_test(y_true=fib_labels,y_score_1=fib_augmented,y_score_2=fib_radiomics)
print(f"Fibrosis DT Rad vs Radiologist{fib_dt_rad_vs_radiologist},\nFibrosis DT AUGRad vs Radiologist{fib_dt_augrad_vs_radiologist},\nFibrosis DT AUGRad vs Rad{fib_dt_augrad_vs_rad}")

###########################################################
#run delong test on Fib in training
fib_training_radiomics=pd.read_csv('fib_training_radiomics.csv',header=None)
fib_labels=fib_training_radiomics.iloc[:,0].to_numpy()
fib_radiologist=fib_training_radiomics.iloc[:,1].to_numpy()
fib_training_aug_rad=pd.read_csv('fib_training_augrad.csv',header=None)

effects,cis,p_values=summary_stats(fib_labels,fib_training_radiomics.iloc[:,2:],fib_training_aug_rad.iloc[:,2:])

# Number of columns to duplicate
x = 100

# Duplicate the columns to create a matrix
duplicated_matrix = np.tile(fib_radiologist, (1, x)).reshape(-1, x)
effects_2,cis_2,p_values_2=summary_stats(fib_labels,fib_training_radiomics.iloc[:,2:],pd.DataFrame(duplicated_matrix))
effects_3,cis_3,p_values_3=summary_stats(fib_labels,fib_training_aug_rad.iloc[:,2:],pd.DataFrame(duplicated_matrix))
print(f"TRAINING Fibrosis DT Rad vs Radiologist{(mean(effects_2),mean(p_values_2))},Fibrosis DT AUGRad vs Radiologist{(mean(effects_3),mean(p_values_3))},Firbsosi DT AUGRad vs Rad{(mean(effects),mean(p_values))}")



'''
