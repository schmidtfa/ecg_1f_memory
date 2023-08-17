#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 13:33:47 2022

@author: schmidtfa
"""
#%% imports
from preprocess_eeg import Preprocessing
from plus_slurm import JobCluster, PermuteArgument
from os import listdir

#%% get jobcluster
job_cluster = JobCluster(required_ram='10G',
                         request_time=4000,
                         request_cpus=2,
                         python_bin='/mnt/obob/staff/fschmidt/conda_cache/envs/ml/bin/python')

OUTDIR = '/mnt/obob/staff/fschmidt/playground/ecg_1f_memory/data/'
base_dir = '/mnt/sinuhe/data_raw/fs_pupils/subject_subject/ds003838'
all_subjects = [subject[4:] for subject in listdir(base_dir) if 'sub' in subject]


# conditions = ['control/high_load', 'control/low_load', 'control/medium_load',
#               'memory/high_load', 'memory/medium_load', 'memory/low_load'
#               ]

conditions = ['baseline', #'control/delay/high', 'control/delay/over', 'control/delay/low', 
              #'memory/baseline', 
              'delay',] #'memory/delay/low', 'memory/delay/high']
#note: for memory trials you can also add /recalled or /forgotten to compare succesful with unsuccseful trials

#%% put in jobs...
job_cluster.add_job(Preprocessing,
                    subject=PermuteArgument(all_subjects),
                    base_dir=base_dir,
                    condition=PermuteArgument(conditions),
                    outdir=OUTDIR,
                    l_pass = None,
                    h_pass = 0.1,
                    notch = False,
                    powerline = 50, #in hz data was recorded in russia 50hz should be fine
                    potato_threshold = 2,
                    #eye_threshold = 0.4,
                    ecg_threshold = 0.4,
                    )
#%% submit...
job_cluster.submit(do_submit=True)
# %%
