from operator import index
from numpy.linalg.linalg import cond
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import dowhy
from dowhy import CausalModel
import statsmodels.formula.api as smf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, SparsePCA

df = pd.read_excel("data_1.xlsx")
df.drop(columns=['Unnamed: 0'], inplace=True)
np_arr = df.values
np_arr1 = np.reshape(np_arr, (-1))
df1 = pd.DataFrame(np_arr1)
df1[['conditions', 'severity_covid_death', 'outcome', 'trazodone']] = df1[0].apply(lambda x: pd.Series(str(x).split('+')))
df1.drop(columns=0, inplace=True)

df1.conditions = df1.conditions.apply(lambda x: str(x).split(','))
df1.conditions = df1.conditions.apply(lambda x: list(map(int, x)))
unique_conditions = list(set(list(itertools.chain.from_iterable(df1.conditions.values))))
unique_conditions.sort()
conditions_matrix = np.zeros((len(df1),len(unique_conditions)), dtype=np.bool)
for i in range(len(df1)):
    encoded_conditions = np.isin(unique_conditions ,df1.loc[i, 'conditions'], assume_unique=True)
    conditions_matrix[i, :] = encoded_conditions 


col_names = []
for i in range(conditions_matrix.shape[1]):
    col_names.append('cond_'+str(i))

df_cond = pd.DataFrame(conditions_matrix.astype(int), 
                        index= df.index, 
                        columns=col_names)

df_w_enc_cond = df1.join(df_cond)
df_w_enc_cond = df_w_enc_cond.rename(columns={'trazodone_bool':'tz'})
