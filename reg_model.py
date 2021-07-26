from operator import index
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import dowhy
import statsmodels.formula.api as smf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

df = pd.read_excel("trazodone.xlsx")
df.drop(columns=['Unnamed: 0'], inplace=True)
df.conditions = df.conditions.apply(lambda x: str(x).split(','))
df.conditions = df.conditions.apply(lambda x: list(map(int, x)))
unique_conditions = list(set(list(itertools.chain.from_iterable(df.conditions.values))))
unique_conditions.sort()
conditions_matrix = np.zeros((len(df),len(unique_conditions)), dtype=np.bool)
for i in range(len(df)):
    encoded_conditions = np.isin(unique_conditions ,df.loc[i, 'conditions'], assume_unique=True)
    conditions_matrix[i, :] = encoded_conditions 

col_names = []
for i in range(conditions_matrix.shape[1]):
    col_names.append('cond_'+str(i))

df_cond = pd.DataFrame(conditions_matrix.astype(int), 
                        index= df.index, 
                        columns=col_names)

df_w_enc_cond = df.join(df_cond)
df_w_enc_cond = df_w_enc_cond.rename(columns={'trazodone_bool':'tz'})

formula = "severity_covid_death ~"

for i in range(conditions_matrix.shape[1]//5):
    formula += f' tz*cond_{i}'
    if i < (conditions_matrix.shape[1]//5 - 1):
        formula += ' +'

reg_model = smf.ols(formula , data= df_w_enc_cond).fit()
reg_model.summary().tables[1]


#Grad Boosting Regression Model
train, test = train_test_split(df_w_enc_cond, test_size = 0.2)
X = np.append(col_names, 'tz')
y = 'severity_covid_death'
gb_model = GradientBoostingClassifier()
gb_model.fit(train[X], train[y])

gb_model.score(test[X], test[y])

