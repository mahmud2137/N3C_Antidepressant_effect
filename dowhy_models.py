from operator import index
from numpy.linalg.linalg import cond
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import dowhy
from tqdm import tqdm
from dowhy import CausalModel
import statsmodels.formula.api as smf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, SparsePCA, IncrementalPCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression, SGDClassifier
from causallib.estimation import IPW 
from causallib.datasets import load_nhefs
from scipy.sparse import csr_matrix

df = pd.read_excel("N3C_data.xlsx")
df.drop(columns=['Unnamed: 0', 'rownum'], inplace=True)

np_arr = df.values
np_arr1 = np.reshape(np_arr, (-1))
df1 = pd.DataFrame(np_arr1)
del df
df1[['conditions', 'severity_covid_death', 'outcome', 'trazodone']] = df1[0].apply(lambda x: pd.Series(str(x).split('+')))
df1.drop(columns=0, inplace=True)
df1.severity_covid_death = df1.severity_covid_death.apply(lambda x: int(x))
df1.outcome = df1.outcome.apply(lambda x: int(x))
df1.trazodone = df1.trazodone.apply(lambda x: int(x))

df1.conditions = df1.conditions.apply(lambda x: str(x).split(','))
df1.conditions = df1.conditions.apply(lambda x: list(map(int, x)))
unique_conditions = list(set(list(itertools.chain.from_iterable(df1.conditions.values))))
unique_conditions.sort()


conditions_matrix = np.zeros((len(df1),len(unique_conditions)), dtype=np.bool)
for i in tqdm(range(len(df1))):
    encoded_conditions = np.isin(unique_conditions ,df1.loc[i, 'conditions'], assume_unique=True)
    conditions_matrix[i, :] = encoded_conditions 

# np.save("condition_matrix.npy", conditions_matrix)

conditions_matrix = np.load("condition_matrix.npy")
conditions_matrix_sp = csr_matrix(conditions_matrix)
col_names = []
for i in range(conditions_matrix.shape[1]):
    col_names.append('cond_'+str(i))

df_cond = pd.DataFrame(conditions_matrix.astype('bool'), 
                        index= df1.index, 
                        columns=col_names)

df1.drop(columns='conditions', inplace=True)

df_w_enc_cond = df1.join(df_cond) 
#Using Causallib
# ipw = IPW(LogisticRegression())
# ipw.fit(df_w_enc_cond.drop(columns=['outcome', 'trazodone']), df_w_enc_cond.trazodone)


#SGD regressor.
train, test = train_test_split(df_w_enc_cond, test_size = 0.2)
X = np.append(col_names, 'trazodone')
y = 'outcome'

sgd_model = SGDClassifier()
sgd_model.fit(train[X], train[y])
sgd_model.score(test[X], test[y])


# with 2D sparse PCA
n_comp = 10
pca = SparsePCA(n_components=n_comp)
conditions_pca = pca.fit_transform(conditions_matrix_sp)
plt.scatter(conditions_pca[0:1000,0], conditions_pca[0:1000,1], marker='.')


# # Increamental PCA
ipca = IncrementalPCA(copy = False, batch_size = 10000, n_components=50)
conditions_ipca = ipca.fit_transform(conditions_matrix)
# Truncated SVD


cols_pca = []
for i in range(n_comp):
        cols_pca.append(f'x{i}')
       

df_cond_pca = pd.DataFrame(conditions_pca, index=df1.index, columns=cols_pca)
df_w_cond_pca = df1.join(df_cond_pca)
df_w_cond_pca.drop(columns=['conditions'], inplace=True)
df_w_cond_pca.trazodone = df_w_cond_pca.trazodone.replace({0:False, 1:True})


graph_list = ''
for i in cols_pca:
        graph_list += f'{i} -> outcome; '

graph = 'digraph {U[label="Unobserved Confounders"]; U -> trazodone; U->outcome ;trazodone -> outcome;'
graph_all = graph + graph_list + '}'
# Model without Dimention reduction
model = CausalModel(
            data = df_w_cond_pca,
            treatment= 'trazodone',
            outcome= 'outcome' ,
            graph = graph_all

)

identified_estimand = model.identify_effect(proceed_when_unidentifiable=False)
print(identified_estimand)

estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.propensity_score_matching", test_significance=True, confidence_intervals=True
)

print(str(estimate))




# Model with Dimention reduction
model_pca = CausalModel(
            data = df_w_cond_pca,
            treatment= 'trazodone',
            outcome= 'outcome',
            graph = 'digraph {U[label="Unobserved Confounders"]; U -> trazodone; U->outcome ;trazodone -> outcome; x1 -> outcome; x2 -> outcome}'

)

identified_estimand = model_pca.identify_effect(proceed_when_unidentifiable=False)
print(identified_estimand)

estimate = model_pca.estimate_effect(identified_estimand,
        method_name="backdoor.linear_regression", test_significance=True, confidence_intervals=True
)

print(str(estimate))

estimate = model_pca.estimate_effect(identified_estimand,
        method_name="backdoor.propensity_score_matching", target_units='att'
)

print(str(estimate))