import pandas as pd
import numpy as np
import json
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import csr_matrix, data
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed


def run_ps(df, X_data, T, y):
    # estimate the propensity score
    ps = LogisticRegression(max_iter=1000 ,C=1e6, n_jobs=1).fit(X_data, df[T]).predict_proba(X_data)[:, 1]
    weight = (df[T]-ps) / (ps*(1-ps)) # define the weights
    return np.mean(weight * y) # compute the ATE

df = pd.read_excel("N3C_full_data.xlsx")
df.drop(columns=[1, '1.1'], inplace=True)
np_arr = df.values
np_arr1 = np.reshape(np_arr, (-1))
df = pd.DataFrame(np_arr1)
df[['conditions', 'age' ,'severity_covid_death', 'outcome', 'zip', 'ethnicity_concept_id', 'gender_concept_id', 'race_concept_id','trazodone', 'amitriptyline', 'fluoxetine', 'citalopram', 'paroxetine', 'venlafaxine', 'vilazodone', 'vortioxetine', 'sertraline', 'bupropion', 'mirtazapine', 'desvenlafaxine', 'doxepin', 'duloxetine', 'escitalopram', 'nortriptyline']] = df[0].apply(lambda x: pd.Series(str(x).split('+')))
df.drop(columns=0, inplace=True)

df.conditions = df.conditions.apply(lambda x: str(x).split(','))
df.conditions = df.conditions.apply(lambda x: list(map(int, x)))




def convert_to_int(x):
    try:
        return int(float(x))
    except ValueError:
        return 0
for col in df.columns[1:]:
    df.loc[:,col] = df.loc[:,col].apply(convert_to_int)

#ignoring data without valid zipcode
df = df[df.zip != 0]
#ignoring children of age less than 13
df = df[df.age > 13]
df.reset_index(drop=True, inplace=True)
anti_depressants = ['trazodone', 'amitriptyline', 'fluoxetine', 'citalopram', 'paroxetine',
       'venlafaxine', 'vilazodone', 'vortioxetine', 'sertraline', 'bupropion',
       'mirtazapine', 'desvenlafaxine', 'doxepin', 'duloxetine',
       'escitalopram', 'nortriptyline']

df['any_anti_depressants'] = df.loc[:, anti_depressants].any(axis='columns').astype('int')

unique_conditions = list(set(list(itertools.chain.from_iterable(df.conditions.values))))
unique_conditions.sort()

# conditions_matrix = np.zeros((len(df),len(unique_conditions)), dtype=np.bool)

# for i in tqdm(range(len(df))):
#     encoded_conditions = np.isin(unique_conditions ,df.loc[i, 'conditions'], assume_unique=True)
#     conditions_matrix[i, :] = encoded_conditions

# np.save("condition_matrix241.npy", conditions_matrix)
# conditions_matrix_sp = csr_matrix(conditions_matrix)
# sparse.save_npz('condition_matrix_sparse.npz', conditions_matrix_sp)

conditions_matrix_sp = sparse.load_npz('condition_matrix_sparse.npz')

df.age = df.age /df.age.max()


neg_controls = ['Fracture_of_bone', 'asthma', 'chronic_kidney_disease', 'disorder_of_nail', 'eczema']
neg_cntrl_snomed_id = [125605004, 195967001, 709044004, 17790008, 43116000]

# Loading concept id to snomed mapping.

df_sn2con = pd.read_excel("snomed_to_concept_id.xlsx")
df_sn2con.drop(columns=['Unnamed: 0'], inplace=True)

np_arr = df_sn2con.values
np_arr1 = np.reshape(np_arr, (-1))
df_sn2con = pd.DataFrame(np_arr1[:-30])

df_sn2con[['snomed_id', 'condition_concept_id']] = df_sn2con[0].apply(lambda x: pd.Series(str(x).split('+')))
df_sn2con.drop(columns=0, inplace=True)


df_sn2con.snomed_id = df_sn2con.snomed_id.astype(int)
df_sn2con.condition_concept_id = df_sn2con.condition_concept_id.apply(lambda x: int(float(x)))

df_sn2con_dict = dict(zip(df_sn2con.snomed_id, df_sn2con.condition_concept_id))

neg_cntrl_cond_id = [df_sn2con_dict[x] for x in neg_cntrl_snomed_id]


neg_cntrl_idx = [np.where(np.array(unique_conditions)==x)[0][0] for x in neg_cntrl_cond_id]

outcome = conditions_matrix_sp[:,neg_cntrl_idx[0]].todense().astype(int)
cols_to_keep = np.delete(np.arange(conditions_matrix_sp.shape[1]), neg_cntrl_idx[0])
cond_mat_sp_wo_outcome = conditions_matrix_sp[:,cols_to_keep]

df.drop(columns=['conditions', 'severity_covid_death'], inplace=True)
categ = ['ethnicity_concept_id', 'gender_concept_id', 'race_concept_id', 'zip']
cont = ['age']

T = 'any_anti_depressants'
Y = 'outcome'
experiments = anti_depressants + ['any_anti_depressants']
prop_score = pd.DataFrame(columns=['prop_score_w_ate_mean', 'ci_95_l', 'ci_95_h'])

data_with_categ = pd.concat([
    df.drop(columns=categ + [i for i in experiments if i != T]), # dataset without the categorical features
    pd.get_dummies(df[categ], columns=categ, drop_first=False)# categorical features converted to dummies
], axis=1)

X = data_with_categ.columns.drop([T,Y])
covariets = sparse.hstack([data_with_categ[X].values, cond_mat_sp_wo_outcome])

np.random.seed(88)
# run 1000 bootstrap samples
bootstrap_sample = 100
ates = Parallel(n_jobs=4)(delayed(run_ps)(data_with_categ.sample(frac=1, replace=True), covariets, T, outcome)
                        for _ in range(bootstrap_sample))
ates = np.array(ates)

print(f"ATE: {ates.mean()}")
print(f"95% C.I.: {(np.percentile(ates, 2.5), np.percentile(ates, 97.5))}")
# prop_score.loc[T,['prop_score_w_ate_mean','ci_95_l','ci_95_h']] = ates.mean(), np.percentile(ates, 2.5), np.percentile(ates, 97.5)
