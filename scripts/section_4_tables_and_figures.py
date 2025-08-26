import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

output = pd.read_csv("../results/output.csv")

# Filter just as we do in section_4_tables_and_figures.R. As of
# 2025-01-28, expected number of rows in df is 37800 * 6 = 226800

#  df <- df %>% filter(func != "rastrigin2d")
#  df <- df %>% filter(v_domain != "bestsample")
#  df <- df %>% filter(v_domain != "svm")
#  df <- df %>% filter(v_domain != "svmplus")

output = output[output['func'] != 'rastrigin2d']
output = output[output['v_domain'] != 'bestsample']
output = output[output['v_domain'] != 'svm']
output = output[output['v_domain'] != 'svmplus']

df = output.pivot(
    index=['func', 'noise', 'sampling', 'sample_sz', 'seed', 'learning'],
    columns='v_domain',
    values='funval_err'
)
df = df.reset_index()[['func', 'sampling', 'sample_sz', 'box', 'ch', 'isofor', 'chplus']]

df = df.rename(columns = {"sampling": "Sampling"})

df['chplus/box'] = df['chplus']/df['box']
df['chplus/ch'] = df['chplus']/df['ch']
df['chplus/isofor'] = df['chplus']/df['isofor']

#remove rastrigin2d, change name of rastrigin10d to simply rastrigin
#df = df[df['func'] != 'rastrigin2d']
#df.loc[df['func']=='rastrigin10d', 'func']='rastrigin'

df.loc[df['Sampling']=='normal_at_min', 'Sampling'] = 'Normal'
df.loc[df['Sampling']=='uniform', 'Sampling'] = 'Uniform'

# plot ch+/box by sampling method
g = sns.FacetGrid(df, col='Sampling', height=3, aspect=1.2, sharex=True, sharey=True, col_order = ['Uniform', 'Normal'])
g.map(sns.kdeplot, 'chplus/box', fill=True, log_scale=10)
g.set_xlabels(r'CH$^+/$ Box')
for samp, ax in g.axes_dict.items():
    ax.axvline(1, color='black', linestyle='--')
    
    filtered_df = df[(df['Sampling']==samp)]
    percent_quantile = np.sum(filtered_df['chplus/box'] < 1) / len(filtered_df)
    percent_quantile_percentage = percent_quantile * 100
    
    ax.text(1 - 0.999, 0.2, f'{percent_quantile_percentage:.2f}%', color='black', ha='center')

#  g.fig.savefig('../results/plots/funvalerr_chplus_over_box.png', dpi = 300,
#                bbox_inches='tight', pad_inches=0)

# plot ch+/ch by sampling method
g = sns.FacetGrid(df, col='Sampling', height=3, aspect=1.2, sharex=True, sharey=True, col_order = ['Uniform', 'Normal'])
g.map(sns.kdeplot, 'chplus/ch', fill=True, log_scale=10, color = 'dimgray')
g.set_xlabels(r'CH$^+/$ CH')
for samp, ax in g.axes_dict.items():
    ax.axvline(1, color='black', linestyle='--')
    
    filtered_df = df[(df['Sampling']==samp)]
    percent_quantile = np.sum(filtered_df['chplus/ch'] < 1) / len(filtered_df)
    percent_quantile_percentage = percent_quantile * 100
    
    ax.text(1 - 0.999, 0.2, f'{percent_quantile_percentage:.2f}%', color='black', ha='center')

g.fig.savefig('../results/figures/section_4_funvalerr_chplus_over_ch.png', dpi = 300,
              bbox_inches='tight', pad_inches=0)

#plot ch+/isofor by sampling method
g = sns.FacetGrid(df, col='Sampling', height=3, aspect=1.2, sharex=True, sharey=True, col_order = ['Uniform', 'Normal'])
g.map(sns.kdeplot, 'chplus/isofor', fill=True, log_scale=10)
g.set_xlabels(r'CH$^+/$ IsoFor')
for samp, ax in g.axes_dict.items():
    ax.axvline(1, color='black', linestyle='--')
    
    filtered_df = df[(df['Sampling']==samp)]
    percent_quantile = np.sum(filtered_df['chplus/isofor'] < 1) / len(filtered_df)
    percent_quantile_percentage = percent_quantile * 100
    
    ax.text(1 - 0.999, 0.2, f'{percent_quantile_percentage:.2f}%', color='black', ha='center')

#  g.fig.savefig('../results/plots/funvalerr_chplus_over_isofor.png', dpi = 300,
#                bbox_inches='tight', pad_inches=0)

# pick the data that generated bad graph
badone = df[(df['func']=='griewank') & (df['Sampling']=='Normal')]
badtwo = df[(df['func']=='peaks') & (df['Sampling']=='Uniform')]
bad = pd.concat([badone, badtwo])
bad

# plot the bad graph separately
g = sns.FacetGrid(bad, col='func', height=3, aspect=1.2, sharex=False, sharey=False)
g.map(sns.kdeplot, 'chplus/box', fill=True, log_scale=10)
for func, ax in g.axes_dict.items():
    ax.axvline(1, color='black', linestyle='--')
    
    filtered_df = bad[(bad['func']==func)]
    percent_quantile = np.sum(filtered_df['chplus/box'] < 1) / len(filtered_df)
    percent_quantile_percentage = percent_quantile * 100
    
    ax.text(1 - 0.9, 0.4, f'{percent_quantile_percentage:.2f}%', color='black', ha='center')
        
axes = g.axes.flatten()
axes[0].set_title("Giewank sampled normally")
axes[1].set_title("Peaks sampled uniformly")
g.set_xlabels(r'CH$^+/$ Box')

#  g.fig.savefig('../results/plots/funvalerr_extra_examples.png', dpi = 300,
#                bbox_inches='tight', pad_inches=0)
