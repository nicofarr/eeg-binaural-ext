from mne.io import read_raw_edf,RawArray
from mne.export import export_raw
from mne import find_events,annotations_from_events,write_events,create_info,find_events,merge_events
from mne.preprocessing import find_eog_events
from mne.viz import plot_events
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from mne import Epochs


import argparse



def find_stim_onsets(Df,key,id):

    stim = Df[key]

    ind = (stim.diff() > 0.)
    D = Df[ind]['sample'].to_numpy()
    
    return np.vstack([D, np.ones_like(D),id*np.ones_like(D)]).T

def addevents(controlfile):

    control = read_raw_edf(controlfile)
    events = control.get_data()

    ### Creating a pandas dataframe for events 

    Df = pd.DataFrame(events.T,columns=control.ch_names)

    Df.index = control.times
    Df['time']  = control.times
    print(control.ch_names)


    Df_crop = Df

    ## estimate the time of the eeg recording in the control file

    Df_crop['reltime'] = Df_crop['time']

    ### Generate some fake signal 

    chnames = ['Cz','STIM']
    chtypes = ['eeg','stim']
    data = np.zeros((2,250*3600*2)) ## fake two hours of signal with one electrode 

    info = create_info(ch_names=chnames,sfreq=250,ch_types=chtypes)

    raw = RawArray(data, info)

    ### Finding the closest sample times 

    Df_crop['sample'] = raw.time_as_index(Df_crop.reltime)

    D = []
    D.append(find_stim_onsets(Df_crop,'stim000',1))
    D.append(find_stim_onsets(Df_crop,'stim001',2))
    D.append(find_stim_onsets(Df_crop,'stim002',3))
    D.append(find_stim_onsets(Df_crop,'stim003',4))
    D.append(find_stim_onsets(Df_crop,'stim004',5))

    D.append(find_stim_onsets(Df_crop,'stim005',6))
    D.append(find_stim_onsets(Df_crop,'stim006',7))
    D.append(find_stim_onsets(Df_crop,'stim007',8))
    D.append(find_stim_onsets(Df_crop,'stim008',9))
    D= np.vstack(D)

    print(np.unique(D[:,2]))

    raw.add_events(D,stim_channel='STIM')
    events = find_events(raw,stim_channel='STIM')

    event_dict = {'NM_NH_1': 1, 'NM_NH_2': 2, 'NM_NH_3': 3,
                'M_NH_1': 4, 'M_NH_2': 5, 'M_NH_3': 6,
                'M_H_1': 7, 'M_H_2': 8 , 'M_H_3': 9}

    merged_events = merge_events(events, [1, 2, 3], 1)
    merged_events = merge_events(merged_events, [4, 5, 6], 2)
    merged_events = merge_events(merged_events, [7, 8, 9], 3)
    event_dict_merged = {'NM, NHT': 1, 'M, NHT': 2, 'MHT': 3}

    print(merged_events)
    assert merged_events.shape[0] == 810


    print(np.unique(merged_events[:,2]))

    return merged_events,raw,event_dict_merged


### write result file

#eventsfile = controlfile[:-4] + '_eve.fif'

#write_events(eventsfile,events,overwrite=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Add events')

    parser.add_argument('--control', type=str,required=True,help='Path to recordcontrol file')                    
    args = parser.parse_args()


    controlfile = args.control

    merged_events,raw,event_dict_merged = addevents(controlfile)

    fig = plot_events(merged_events, sfreq=raw.info['sfreq'],
                            first_samp=raw.first_samp, event_id=event_dict_merged)



