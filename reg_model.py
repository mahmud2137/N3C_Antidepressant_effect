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


def pred_elasticity(m, df, t="tz"):
    return df.assign(**{
        "pred_elast": m.predict(df.assign(**{t:df[t]+1})) - m.predict(df)
    })

pred_elas = pred_elasticity(reg_model, test)

bands_df = pred_elas.assign(
    elast_band = pd.qcut(pred_elas["pred_elast"], 2),
    pred_death = reg_model.predict(pred_elas),
    pred_band = pd.qcut(reg_model.predict(pred_elas), 2),
)

bands_df

g = sns.FacetGrid(bands_df, col="elast_band")
g.map_dataframe(sns.regplot, x="tz", y="severity_covid_death")
g.set_titles(col_template="Elast. Band {col_name}");

pca = SparsePCA(n_components=2)
conditions_pca = pca.fit_transform(conditions_matrix)
plt.scatter(conditions_pca[:,0], conditions_pca[:,1], marker='.')

df_cond_pca = pd.DataFrame(conditions_pca, index=df.index, columns=['x1', 'x2'])
df_w_cond_pca = df.join(df_cond_pca)
df_w_cond_pca = df_w_cond_pca.rename(columns={'trazodone_bool':'tz'})
df_w_cond_pca.drop(columns=['conditions', 'Per'], inplace=True)
df_w_cond_pca
#dowhy Causal Model
model = CausalModel(
            data = df_w_cond_pca,
            treatment= 'tz',
            outcome= 'severity_covid_death' ,
            graph = 'digraph {U[label="Unobserved Confounders"]; U -> tz; U->severity_covid_death ;tz -> severity_covid_death;}'

)

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)

estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.linear_regression", test_significance=True, confidence_intervals=True
)

print(str(estimate))