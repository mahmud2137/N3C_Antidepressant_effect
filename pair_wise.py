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

# plt.hist(df["outcome"], bins=2, alpha=0.3, label="All")
# plt.hist(df.query("any_anti_depressants==0")["outcome"], bins=2, alpha=0.3, color="C2", label='untreated')
# plt.hist(df.query("any_anti_depressants==1")["outcome"], bins=2, alpha=0.3, color="C3", label= 'treated')
# plt.legend();
 

def run_ps(df, X_data, T, y):
    # estimate the propensity score
    ps = LogisticRegression(max_iter=1000 ,C=1, n_jobs=-1).fit(X_data, df[T]).predict_proba(X_data)[:, 1]
    weight = (df[T]-ps) / (ps*(1-ps)) # define the weights
    return np.mean(weight * df[y]) # compute the ATE

# df1 = df.drop(columns=anti_depressants)

df.drop(columns=['conditions', 'severity_covid_death'], inplace=True)
categ = ['ethnicity_concept_id', 'gender_concept_id', 'race_concept_id', 'zip']
cont = ['age']

# T = 'any_anti_depressants'
Y = 'outcome'
experiments = anti_depressants + ['any_anti_depressants']

prop_score_mean = pd.DataFrame(columns=anti_depressants)
prop_score_ci_L = pd.DataFrame(columns=anti_depressants)
prop_score_ci_U = pd.DataFrame(columns=anti_depressants)

for T_a, T_b in tqdm(itertools.combinations(anti_depressants, 2)):

    df_pair = df[df[T_a] ^ df[T_b] == 1]
    cond_matx_sp = conditions_matrix_sp[df_pair.index, :]
    T = T_a


    data_with_categ = pd.concat([
        df_pair.drop(columns=categ + [i for i in anti_depressants if i != T]), # dataset without the categorical features
        pd.get_dummies(df_pair[categ], columns=categ, drop_first=False)# categorical features converted to dummies
    ], axis=1)
    X = data_with_categ.columns.drop([T ,Y])

    covariets = sparse.hstack([data_with_categ[X].values, cond_matx_sp])

    np.random.seed(88)
    # run 1000 bootstrap samples
    bootstrap_sample = 200
    ates = Parallel(n_jobs=-1)(delayed(run_ps)(data_with_categ.sample(frac=1, replace=True), covariets, T, Y)
                            for _ in range(bootstrap_sample))
    ates = np.array(ates)

    print(f"ATE: {ates.mean()}")
    print(f"95% C.I.: {(np.percentile(ates, 2.5), np.percentile(ates, 97.5))}")
    ate_mean = ates.mean()
    ate_ci_L = np.percentile(ates, 2.5)
    ate_ci_U = np.percentile(ates, 97.5)

    prop_score_mean.loc[T_a,T_b] = ate_mean
    prop_score_ci_L.loc[T_a,T_b] = ate_ci_L
    prop_score_ci_U.loc[T_a,T_b] = ate_ci_U

prop_score_mean.to_csv('ate_prop_score_pair_mean.csv')
prop_score_ci_L.to_csv('ate_prop_score_pair_ci_L.csv')
prop_score_ci_U.to_csv('ate_prop_score_pair_ci_U.csv')