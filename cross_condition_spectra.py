#%%
import pandas as pd
import mne
import scipy.signal as dsp
import joblib
from pathlib import Path
from os.path import join
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pingouin as pg
from fooof import FOOOFGroup
from fooof.utils.params import compute_knee_frequency
from scipy.stats import zscore
import pymc as pm
import arviz as az
import bambi as bmb

from fooof.bands import Bands
from fooof.objs.utils import average_fg
from fooof.sim import gen_power_spectrum

import matplotlib as mpl
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

sns.set_style('ticks')
sns.set_context('poster')

#%%

conditions = ['control_baseline', 'memory_baseline', 'memory_delay_over', 'memory_delay_low', 'memory_delay_high']


all_subject_data = []

for cur_condition in conditions:

    all_subjects = list(Path(join('/mnt/obob/staff/fschmidt/playground/ecg_1f_memory/data', cur_condition)).glob('*.dat'))

    for subject in all_subjects:

        cur_sub = joblib.load(subject)
        cur_data = cur_sub['data_brain'].to_data_frame()

        cur_df = pd.DataFrame({
                    'ecg': cur_data['ECG'],
                    })
        cur_df['subject_id'] = cur_sub['subject_id']
        cur_df['condition'] = cur_condition

        all_subject_data.append(cur_df)
df_raw = pd.concat(all_subject_data)

df_raw['condition'] = df_raw['condition'].replace({
                'control_baseline': 'baseline',
                'memory_baseline': 'baseline',
                'memory_delay_over': '13', 
                'memory_delay_low': '5', 
                'memory_delay_high': '9'})



#%% cut data appropriately
df_list_cut = []

for cur_sub in df_raw['subject_id'].unique():
    
    cur_df = df_raw.query(f'subject_id == "{cur_sub}"')

    baseline = cur_df.query('condition == "baseline"')
    overload = cur_df.query('condition == "13"')
    highload = cur_df.query('condition == "9"')
    lowload = cur_df.query('condition == "5"')
    
    shortest = np.min([overload.shape[0], highload.shape[0], lowload.shape[0], baseline.shape[0]]) 
    df_list_cut.append(pd.concat([overload[:shortest], highload[:shortest], lowload[:shortest], baseline[:shortest]]))

    df_raw_cut = pd.concat(df_list_cut)

#%%
all_subject_data = []

for condition in df_raw_cut['condition'].unique():
        
      df_cond = df_raw_cut.query(f'condition == "{condition}"')

      for subject in df_cond['subject_id'].unique():

        cur_sub = df_cond.query(f'subject_id == "{subject}"')

        fs = 500
        welch_settings = {'fs': fs,
                          'nperseg': fs*3,
                          'average':'median',
                          'detrend':'linear'  
                            }

        freq, psd_ecg = dsp.welch(cur_sub['ecg'].to_numpy(),  **welch_settings)

        cur_df = pd.DataFrame({'Frequency (Hz)': freq,
                               'ecg': psd_ecg,
                               })
        cur_df['subject_id'] = subject
        cur_df['condition'] = condition

        all_subject_data.append(cur_df)

df_cm = pd.concat(all_subject_data)

df_cm = df_cm.query('`Frequency (Hz)` <= 245') #make sure that i am not fitting in a 

df_cm.to_csv('./data/psd_df_ecg_memory.csv')



#%%
fg = FOOOFGroup(max_n_peaks=2, 
                peak_width_limits=(1, 6), 
                aperiodic_mode='knee')
bands = Bands({'delta': [0, 2],
               'theta': [4, 8],
               'alpha': [7, 16]
               })

freq = np.unique(df_cm['Frequency (Hz)'])


df_info = (df_cm.query('condition == "13"')
                .drop_duplicates('subject_id')[['subject_id', 'condition']]
                .reset_index())


#%%

fooof_list = []

for condition in df_cm['condition'].unique():

    cond_data = (df_cm.query(f'condition == "{condition}"')
                    .pivot(columns='Frequency (Hz)', values='ecg', index='subject_id')
                    .to_numpy())
    
    fg.fit(freq, cond_data)

    df_hl = pd.DataFrame(fg.get_params('aperiodic_params'), columns = ['offset', 'knee', 
                                                                       'exponent'])
    df_hl['condition'] = condition
    df_hl['subject_id'] = [int(id) for id in df_info['subject_id']]
    df_hl['r2'] = fg.get_params('r_squared')

    fooof_list.append(df_hl)

#%%
df_spec = pd.concat(fooof_list)

#%%
fig, ax = plt.subplots(figsize=(6,4))

xticklabels = ['over load', 'high load', 'low load', 'baseline']

df_spec['slope'] = df_spec['exponent'] * -1

sns.pointplot(df_spec, y='condition', x='slope', errorbar='se',
               color='#333333', ax=ax)
ax.set_yticklabels(xticklabels, )#rotation = 90,)

sns.despine()
#fig.savefig('./results/exponent_x_load.svg')


#%%

fit_kwargs = {
    'draws': 4000,
    'tune': 2000,
    'chains': 4,
    'target_accept': 0.9,
    'idata_kwargs' : {'log_likelihood': True}, #needed for loo and waic
}


#%%
lvl = ['baseline', '5', '9', '13']
md_1_s = bmb.Model('scale(slope) ~ 1 + C(condition, levels=lvl) + (1|subject_id)',
                      data=df_spec)
mdf_1_s = md_1_s.fit(**fit_kwargs)

#md_1.predict(mdf_1)
#md_1.predict(mdf_1, kind='pps')
az.summary(mdf_1_s)
#%%
lvl = ['baseline', '5', '9', '13']
md_1 = bmb.Model('slope ~ 1 + C(condition, levels=lvl) + (1|subject_id)',
                      data=df_spec)
mdf_1 = md_1.fit(**fit_kwargs)

#md_1.predict(mdf_1)
md_1.predict(mdf_1, kind='pps')

#%%
ridge_df = pd.DataFrame({'5':  mdf_1.posterior['C(condition, levels=lvl)'][:,:,0].to_numpy().flatten(),
                         '9':  mdf_1.posterior['C(condition, levels=lvl)'][:,:,1].to_numpy().flatten(),
                         '13': mdf_1.posterior['C(condition, levels=lvl)'][:,:,2].to_numpy().flatten()}).melt(var_name='n digits to recall',
                                                                                                                value_name='spectral slope (Δ Baseline)')
#%%
def plot_ridge(df, variable_name, values, pal, plot_order=None, xlim=(-1,1), aspect=15, height=0.5):

      '''Pretty ridge plot function in python. This is based on code from https://seaborn.pydata.org/examples/kde_ridgeplot.html'''

      sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

      plotting_kwargs = {'row':variable_name, 
                         'hue': variable_name, 
                         'aspect': aspect,
                         'height': height,
                         'palette': pal}

      if plot_order is not None:
            plotting_kwargs.update({'row_order': plot_order,
                                    'hue_order': plot_order})
      
      g = sns.FacetGrid(df, **plotting_kwargs)

      g.map(sns.kdeplot, values,
            bw_adjust=.5, clip_on=False,
            fill=True, alpha=1, linewidth=1.5)
      g.map(sns.kdeplot, values, clip_on=False, color="w", lw=2, bw_adjust=.5)

      # passing color=None to refline() uses the hue mapping
      g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

      # Define and use a simple function to label the plot in axes coordinates
      def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color=color,
                  ha="left", va="center", transform=ax.transAxes)

      g.map(label, values)

      # Set the subplots to overlap
      g.figure.subplots_adjust(hspace=-.25)

      # Remove axes details that don't play well with overlap
      g.set_titles("")
      g.set(xlim=xlim)
      g.set(yticks=[], ylabel="")
      g.despine(bottom=True, left=True)

      return g

#%%
plot_order = ['5', '9', '13']
pal = ['#fec44f', '#d95f0e', '#993404']

g = plot_ridge(ridge_df, 
           variable_name='n digits to recall',
           values = 'spectral slope (Δ Baseline)',
           pal=pal,#'Reds_r',
           plot_order=plot_order,
           xlim=(0, 0.35),
           aspect=6,
           height=0.5)

g.figure.savefig('./results/condition_diff_ridge.svg')



#%% scale data
df_spec['slope_z'] = zscore(df_spec['slope'])

