from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import sklearn.metrics
import roc
import pandas as pd

###########################################################
#run delong test on the inflammation

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
for n in inf_dt_augrad_vs_radiologist:
    print(n,'\n')

print(f"Inflammation DT Rad vs Radiologist{inf_dt_rad_vs_radiologist},\nInflammation DT AUGRad vs Radiologist{inf_dt_augrad_vs_radiologist},\nInflammation DT AUGRad vs Rad{inf_dt_augrad_vs_rad}")

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



