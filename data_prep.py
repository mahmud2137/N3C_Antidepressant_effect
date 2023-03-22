import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("N3C_data.xlsx")
df.drop(columns=['Unnamed: 0'], inplace=True)

ate_prop = pd.read_csv('ate_prop_score.csv')
ate_prop['ci'] = ate_prop.ci_95_h - ate_prop.ci_95_l
plt.figure(figsize=(10,6))
ax = sns.boxplot(data=ate_prop, x='Antidepressants', y='prop_score_w_ate_mean', color='green')
ax.errorbar(data=ate_prop, x='Antidepressants', y='prop_score_w_ate_mean', yerr='ci',ls='', color='black')
ax.tick_params(axis='x', labelrotation = 45, labelsize = 14)
ax.set_ylabel('ATE', fontsize = 16)
ax.set_xlabel('Antidepressants', fontsize = 16)
ax.set_title('ATE on antidepressants', fontsize = 16)
plt.tight_layout()
plt.savefig('ATE_on_Antidepressants.png', dpi=300)
plt.show()

ate_pair_mean = pd.read_csv('ate_prop_score_pair_mean.csv')
ate_pair_L = pd.read_csv('ate_prop_score_pair_ci_L.csv')
ate_pair_U = pd.read_csv('ate_prop_score_pair_ci_U.csv')

first_ad = ate_pair_mean.columns[2:]
first_ad[14]
fig, axes = plt.subplots(15,1, figsize = (8,20), sharex=True)
for i in range(15):
    ate_pair_mean['ci'] = ate_pair_U[first_ad[14-i]] - ate_pair_L[first_ad[14-i]]
    sns.barplot(ax=axes[i],data=ate_pair_mean, x='Antidepressants', y=first_ad[14-i], color='blue', alpha = 0.4)
    axes[i].errorbar(data=ate_pair_mean, x='Antidepressants', y=first_ad[14-i], yerr='ci',ls='', color='black' )
    axes[i].tick_params(axis='x', labelrotation = 45)

fig.savefig('pairwise_ate.png')
plt.show()

ate_pair_L

ate_pair_L.set_index('Antidepressants', inplace=True)
ate_pair_U.set_index('Antidepressants', inplace=True)
ate_pair_mult = ate_pair_L.T * ate_pair_U.T
mask_sig = ate_pair_mult > 0
mask_nsig = ate_pair_mult <= 0

ate_pair_mean.set_index('Antidepressants', inplace=True)
#HeatMap
# ate_pair_mean.drop(columns=['ci'], inplace=True)
f, ax = plt.subplots(figsize=(12, 10))
ate_pair_mean.T
ate_pair_mult
sns.heatmap(ate_pair_mean.T, annot=True, fmt=".2f", linewidths=.5, ax=ax,cmap='coolwarm', mask=mask_nsig, annot_kws={"weight": "bold"})
sns.heatmap(ate_pair_mean.T, mask=mask_sig, cbar=False, annot=True, ax=ax, cmap='coolwarm',vmax=0.65, vmin=-0.65)

plt.tight_layout()
f.savefig('ate_heatmap.png', dpi=300)
plt.show()

#Neg Control Graphs


ate_neg_ctl_mean = pd.read_csv('neg_control_results/ate_prop_score_neg_cntl_mean.csv')
ate_neg_ctl_L = pd.read_csv('neg_control_results/ate_prop_score_neg_cntl_ci_L.csv')
ate_neg_ctl_U = pd.read_csv('neg_control_results/ate_prop_score_neg_cntl_ci_U.csv')

ate_neg_ctl_mean = ate_neg_ctl_mean.set_index('Neg_Control').T
ate_neg_ctl_mean = ate_neg_ctl_mean.reset_index().rename(columns={'index':'Antidepressants'})

ate_neg_ctl_L = ate_neg_ctl_L.set_index('Neg_Control').T
ate_neg_ctl_L = ate_neg_ctl_L.reset_index().rename(columns={'index':'Antidepressants'})

ate_neg_ctl_U = ate_neg_ctl_U.set_index('Neg_Control').T
ate_neg_ctl_U = ate_neg_ctl_U.reset_index().rename(columns={'index':'Antidepressants'})
neg_cntls = ate_neg_ctl_mean.columns[1:]

# ate_neg_ctl_U['asthma'] - ate_neg_ctl_L['asthma']

fig, axes = plt.subplots(len(neg_cntls),1, figsize = (8,12), sharex=True, sharey=True)
for i, n in enumerate(neg_cntls):
    ate_neg_ctl_mean['ci'] = ate_neg_ctl_U[n] - ate_neg_ctl_L[n]
    sns.barplot(ax=axes[i],data=ate_neg_ctl_mean, x='Antidepressants', y=n, color='blue', alpha = 0.4)
    axes[i].axhline(0)
    axes[i].errorbar(data=ate_neg_ctl_mean, x='Antidepressants', y=n, yerr='ci',ls='', color='black' )
    axes[i].tick_params(axis='x', labelrotation = 45)

plt.tight_layout()
fig.savefig('neg_cntl.png', dpi=300)
plt.show()