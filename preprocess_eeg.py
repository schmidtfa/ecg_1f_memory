#%%
from plus_slurm import Job
#%%
import os
from os import listdir
from os.path import join

from mne_bids import BIDSPath, read_raw_bids

import mne
import pandas as pd
import numpy as np
from autoreject import Ransac  # noqa
import pyriemann
import joblib
from scipy.signal import welch
import h5py

from fooof import FOOOFGroup
import pandas as pd

#%%

class Preprocessing(Job):

    def run(self, 
            subject,
            base_dir,
            outdir,
            condition,
            l_pass = 145,
            h_pass = 0.1,
            notch = False,
            min_time = 0,
            max_time=3.,
            powerline = 50, #in hz
            #retention=False,
            potato_threshold = 2,
            #eye_threshold = 0.4,
            ecg_threshold = 0.4,
            ):
    
        # %% read raw data
        # base_dir = '/mnt/sinuhe/data_raw/fs_pupils/subject_subject/ds003838'
        # all_subjects = [subject[4:] for subject in listdir(base_dir) if 'sub' in subject]
        # subject = '032'
        # %% read data
        bids_path_eeg = BIDSPath(root=base_dir, subject=subject, task='memory', datatype='eeg')
        raw = read_raw_bids(bids_path=bids_path_eeg, verbose=False)
        # set an eeg montage
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)

        #as mne bids doesnt accept ecg yet i do it by hand
        f_path_ecg = join(base_dir, f'sub-{subject}', 'ecg')
        f_name_ecg = f'sub-{subject}_task-memory_ecg.set'
        raw_ecg = mne.io.read_raw_eeglab(join(f_path_ecg, f_name_ecg))

        #%% read pupil data
        #f_path_pup = join(base_dir, f'sub-{subject}', 'pupil')
        #f_name_pup = f'sub-{subject}_task-memory_pupil.tsv'
        #raw_pup = pd.read_table(join(f_path_pup, f_name_pup))

        # %% reject bad channels
        epo4ransac = mne.make_fixed_length_epochs(raw, duration=4)
        epo4ransac.load_data()
        ransac = Ransac(verbose=True)
        ransac.fit(epo4ransac)
        bad_chs = ransac.bad_chs_

        print(f'RANSAC detected the following bad channels: {bad_chs}')

        raw.info['bads'] = bad_chs
            
        print('Bad channels are repaired using interpolation')
        raw.interpolate_bads() #Interpolate bad channels

        #forcing mne to do what i want(checked the time should be fine)
        raw.add_channels([raw_ecg], force_update_info=True)
        raw.set_channel_types({'ECG': 'ecg',
                               'PPG': 'ecg'})

        # %% filter data
        raw.filter(l_freq=h_pass, h_freq=l_pass)

        if notch == True:
            nyquist = raw.info['sfreq'] / 2
            print(f'Running notch filter using {powerline} Hz steps. Nyquist is {nyquist}')
            raw.notch_filter(np.arange(powerline, nyquist, powerline), filter_length='auto', phase='zero')


        raw = raw.set_eeg_reference(ref_channels='average',
                                    projection=False, verbose=False)

        # %% run ica
        #Get events and annotate raw data
        f_path_events = join(base_dir, f'sub-{subject}', 'eeg')
        f_name_events = f'sub-{subject}_task-memory_events.tsv'
        df_triggers = pd.read_csv(join(f_path_events, f_name_events), sep='\t')

        #trigger definition (i could do this prettier but i really dont care)
        event_list = []
        for idx in df_triggers.index:
            trigger = df_triggers.loc[idx]
            if trigger['value'] != 'boundary':

                if trigger['value'].startswith('50'):
                    event_key = 'control'
                    if int(trigger['value'][2:4]) == 1:
                        event_key += 'baseline'
                        latency = trigger['sample'] - 3000

                    if trigger['value'].endswith('13') and int(trigger['value'][2:4]) == 13:
                        event_key += '/delay/over'
                    elif trigger['value'].endswith('09') and int(trigger['value'][2:4]) == 9:
                        event_key += '/delay/high'
                    elif trigger['value'].endswith('05') and int(trigger['value'][2:4]) == 5:
                        event_key += '/delay/low'
                    latency = trigger['sample']
                elif trigger['value'].startswith('60'):
                    event_key = 'memory'
                    if int(trigger['value'][2:4]) == 1:
                        event_key += 'baseline'
                        latency = trigger['sample'] - 3000

                    if trigger['value'][-3:-1] == '13' and int(trigger['value'][2:4]) == 13:
                        event_key += '/delay/over'
                    elif trigger['value'][-3:-1] == '09' and int(trigger['value'][2:4]) == 9:
                        event_key += '/delay/over'
                    elif trigger['value'][-3:-1] == '05' and int(trigger['value'][2:4]) == 5:
                        event_key += '/delay/over'
                    latency = trigger['sample']
                
                event_list.append(pd.DataFrame({'latency': latency,
                                                'duration': 0,
                                                'trigger': event_key}, index=[0]))
                
                
                 
        event_df = pd.concat(event_list)


        #%% this is super ugly but it works (not event ids might differ across subjects, but as the analysis is within subject it shouldnt matter)
        
        event_map = dict(zip(event_df['trigger'].unique(), np.arange(len(event_df['trigger'].unique()))))
        event_df['trigger'] = event_df['trigger'].replace(event_map)
        event_data = event_df.to_numpy()


        # %% split data in epochs

        epochs = mne.Epochs(raw, event_data, tmin=min_time, 
                            tmax=max_time, event_id=event_map, 
                            baseline=(0,0), preload=True)[condition]

        epochs.resample(500)


        #%% read subject specific meta data
        df = pd.read_csv(join(base_dir, 'participants.tsv'), sep='\t')
        s_id = 'sub-' + subject
        cur_df = df.query(f'participant_id == "{s_id}"')
        
        #%% read behavioral data
        data = {
                'data_brain': epochs,
                'subject_id': subject,
                'condition': condition.replace('/', '_'),
                'age': cur_df['age'],
                'sex': cur_df['sex'],
                'eeg_excluded': cur_df['EEG_excluded'],
                'ecg_excluded': cur_df['ECG_excluded'],
                'behavior_excluded': cur_df['behavior_excluded'],
                'subj_effort': cur_df[['NASA_effort_1', 'NASA_effort_2', 'NASA_effort_3']].mean(axis=1),
                'subj_perf': cur_df[['NASA_perf_1', 'NASA_perf_2', 'NASA_perf_3']].mean(axis=1),
                'subj_temp_demand': cur_df[['NASA_speed_1', 'NASA_speed_2', 'NASA_speed_3']].mean(axis=1),
                'subj_mental_demand': cur_df[['NASA_cog_1', 'NASA_cog_2', 'NASA_cog_3']].mean(axis=1),
                'subj_phys_demand': cur_df[['NASA_phys_1', 'NASA_phys_2', 'NASA_phys_3']].mean(axis=1),}

        #% add behavioral data if present 
        df_beh = pd.read_csv(join(base_dir, f'sub-{subject}', 'beh', f'sub-{subject}_task-memory_beh.tsv'), sep='\t') 
        scores_by_cond = df_beh.groupby('condition').mean().reset_index()[['condition', 'NCorrect']]

        condition_info_list = condition.split('/')
        if condition_info_list[0] == 'memory':
            if condition_info_list[1] == 'high_load':
                data.update({'performance': scores_by_cond.query('condition == 13')['NCorrect']})
            elif condition_info_list[1] == 'medium_load':
                data.update({'performance': scores_by_cond.query('condition == 9')['NCorrect']})
            elif condition_info_list[1] == 'low_load':
                data.update({'performance': scores_by_cond.query('condition == 5')['NCorrect']})

        save_string = subject + ".dat"
        
        real_outdir = join(outdir, condition.replace('/', '_'))
        if not os.path.isdir(real_outdir):
            os.makedirs(real_outdir)

        joblib.dump(data, join(real_outdir, save_string))

# %%
