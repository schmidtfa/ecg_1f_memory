
#%%
import pandas as pd
import mne
import scipy.signal as dsp
import joblib
from os import listdir

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

all_subjects = listdir('/mnt/obob/staff/fschmidt/playground/ecg_1f_memory/data/memory_baseline')
base_dir = '/mnt/sinuhe/data_raw/fs_pupils/subject_subject/ds003838'

#%%

all_subs = []

for subject in all_subjects:


    df_beh = pd.read_csv(join(base_dir, f'sub-{subject[:3]}', 'beh', f'sub-{subject[:3]}_task-memory_beh.tsv'), sep='\t') 
    scores_by_cond = df_beh.groupby('condition').mean().reset_index()[['condition', 'NCorrect', 'partialScore']]
    scores_by_cond['subject_id'] = subject[:3]
    all_subs.append(scores_by_cond)

# %%
df_beh_cmb = pd.concat(all_subs)
# %%
df_beh_cmb.to_csv('./data/df_behav.csv')
# %%
sns.catplot(df_beh_cmb, x='condition', y='NCorrect', kind='point')
# %%
