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

df_raw.replace({'control_baseline': 'baseline',
                'memory_baseline': 'baseline',
                'memory_delay_over': 'delay', 
                'memory_delay_low': 'delay', 
                'memory_delay_high': 'delay'}, inplace=True)

#%% adjust data length (I.e. get same amount of trials in delay and baseline)
df_list_cut = []

for cur_sub in df_raw['subject_id'].unique():
    
    cur_df = df_raw.query(f'subject_id == "{cur_sub}"')

    df_bl = cur_df.query('condition == "baseline"')
    df_dl = cur_df.query('condition == "delay"')

    df_list_cut.append(pd.concat([df_bl[:df_dl.shape[0]], df_dl]))

df_raw_cut = pd.concat(df_list_cut)

#%% do welch

all_subject_data = []

for condition in df_raw_cut['condition'].unique():
        
      df_cond = df_raw_cut.query(f'condition == "{condition}"')

      for subject in df_cond['subject_id'].unique():

        cur_sub = df_cond.query(f'subject_id == "{subject}"')

        fs = 500
        welch_settings = {'fs': fs,
                          'nperseg': fs*3, #used to be 2
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
#%%
df_cm = df_cm.query('`Frequency (Hz)` <= 245') # used to be 245Hz make sure that i am not fitting in a 

df_cm.to_csv('./data/psd_df_ecg.csv')

#%%

fig, ax = plt.subplots(figsize=(4, 6))
sns.lineplot(data=df_cm, y='ecg', x='Frequency (Hz)', 
             palette='deep', errorbar='se',
             hue='condition', ax=ax)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylabel('Power')

sns.despine()
fig.savefig('./results/raw_spectra_ecg.svg')

#%% plot single subject data
g = sns.FacetGrid(df_cm, col='subject_id', col_wrap=4, aspect=0.75,)
g.map_dataframe(sns.lineplot, y='ecg', x='Frequency (Hz)',
                 hue='condition',
                ).set(yscale='log', xscale='log')
g.add_legend()

# %%
delay = (df_cm.query(f'condition == "delay"')
                    .copy()
                    .pivot(columns='Frequency (Hz)', values='ecg', index='subject_id')
                    .to_numpy())

baseline = (df_cm.query(f'condition == "baseline"')
                    .copy()
                    .pivot(columns='Frequency (Hz)', values='ecg', index='subject_id')
                    .to_numpy())

#%%
df_info = (df_cm.query('condition == "delay"')
                .drop_duplicates('subject_id')[['subject_id', 'condition']]
                .reset_index())


#%%
fg = FOOOFGroup(max_n_peaks=2, 
                peak_width_limits=(1, 6), 
                aperiodic_mode='knee')
bands = Bands({'delta': [0, 2],
               'theta': [4, 8],
               'alpha': [8, 14]
               })
freq = np.unique(df_cm['Frequency (Hz)'])

fg.fit(freq, delay)

df_dl = pd.DataFrame(fg.get_params('aperiodic_params'), columns = ['offset', 'knee', 
                                                                   'exponent'])
df_dl['condition'] = 'Delay'
df_dl['subject_id'] = df_info['subject_id']
df_dl['r2'] = fg.get_params('r_squared')

fm_dl = average_fg(fg, bands=bands,  avg_method='median')

# %%
fg.fit(freq, baseline,)
df_bl = pd.DataFrame(fg.get_params('aperiodic_params'), columns = ['offset', 'knee', 
                                                                   'exponent'])
df_bl['condition'] = 'Baseline'
df_bl['subject_id'] =  df_info['subject_id']
df_bl['r2'] = fg.get_params('r_squared')

fm_bl = average_fg(fg, bands=bands, avg_method='median')

#%%
df_load = pd.concat([df_bl,
                     df_dl])

df_load['knee_freq'] = compute_knee_frequency(df_load['knee'], df_load['exponent'])

#%%
pal = sns.color_palette('deep')

fig, axes = plt.subplots(2,1)
axes = axes.reshape(-1)

for ix, (ax, cur_knee) in enumerate(zip(axes, df_load['condition'].unique())):
    sns.histplot(df_load.query(f'condition == "{cur_knee}"')['knee_freq'], 
                 ax=ax, color=pal[ix])
    ax.set_xlim(0, 14)
    ax.set_xlabel('Frequency (Hz)')

fig.tight_layout()
fig.set_size_inches(3,6)
sns.despine()
#fig.savefig('./results/knee_freq_hist.svg')

#%% get the average knee freq and plot slope from there on

print(f'The average knee frequency is at {df_load.knee_freq.mean()}')
#%%

freqs, pwd_bl = gen_power_spectrum(freq_range=[0.1, 245], 
                   aperiodic_params=fm_bl.get_params('aperiodic_params'),
                   freq_res=1/3,
                   nlv=0,
                   periodic_params=[0,0,0],
                   )

#%%
freqs, pwd_dl = gen_power_spectrum(freq_range=[0.1, 245], 
                   aperiodic_params=fm_dl.get_params('aperiodic_params'),
                   nlv=0,
                   freq_res=1/3,
                   periodic_params=[0,0,0]
                   )

#%%
pal = sns.color_palette('deep')


fig, ax = plt.subplots(figsize=(4, 6))
ax.loglog(freqs[freqs > 5], pwd_bl[freqs > 5], label='Baseline', color=pal[0])
ax.loglog(freqs[freqs > 5], pwd_dl[freqs > 5], label='Delay', color=pal[1])
ax.set_ylabel('Power')
ax.set_xlabel('Frequency (Hz)')

plt.legend()
sns.despine()
fig.savefig('./results/aperiodic_model_ecg.svg')

#%%
fm_bl.print_results()

#%%
fm_dl.print_results()



# %%
fig, ax = plt.subplots(figsize=(4, 7))

df_load['spectral slope'] = df_load['exponent'] * -1

sns.swarmplot(data=df_load, x='condition', y='spectral slope', hue='condition', 
              ax=ax, size=10, alpha=0.5, palette='deep')

sns.pointplot(data=df_load, x='condition', y='spectral slope', errorbar='se' ,
              ax=ax, markers='_', hue='condition', scale=2, palette='deep')

ax.get_legend().remove()
ax.set_xlabel('')
sns.despine()
#fig.savefig('./results/slope_comp_ecg.svg')
# %% compare this using every sensible test (just to be sure)

#parametric
pg.ttest(df_load.query('condition == "Baseline"')['spectral slope'], 
          df_load.query('condition == "Delay"')['spectral slope'], paired=True)

#%% non parametric
pg.wilcoxon(df_load.query('condition == "Baseline"')['spectral slope'], 
          df_load.query('condition == "Delay"')['spectral slope'],)
# %% bayesian
# zscore to get standardized betas
df_load['slope_z'] = zscore(df_load['spectral slope'])
df_load['slope'] = df_load['spectral slope']
#%%
fit_kwargs = {
    'draws': 4000,
    'tune': 2000,
    'chains': 4,
    'target_accept': 0.9,
    'idata_kwargs' : {'log_likelihood': True}, 
}

#scale to get standardized betas    
md_1 = bmb.Model('scale(slope) ~ 1 + condition + (1|subject_id)', data=df_load)
mdf_1 = md_1.fit(**fit_kwargs)

sum1 = az.summary(mdf_1)

md_1.predict(mdf_1, kind='pps')
#%% this model fails to fit
md_2= bmb.Model('scale(slope) ~ 1 + condition + (1 + condition|subject_id)', data=df_load)
mdf_2 = md_2.fit(**fit_kwargs)

sum2 = az.summary(mdf_2)

md_2.predict(mdf_2, kind='pps')
#%%
#Compare all models:
models = {
    "random_intercept_fixed_slope": az.convert_to_inference_data(mdf_1),
    "random_intercept_random_slope": az.convert_to_inference_data(mdf_2),
}
df_compare = az.compare(models, ic='waic')
df_compare

#%%
az.plot_compare(df_compare, insample_dev=False);
# %%
