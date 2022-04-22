from mne.io import read_raw_edf,RawArray
from mne.export import export_raw
from mne import find_events,annotations_from_events,write_events,create_info,find_events
from mne.preprocessing import find_eog_events
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from mne import Epochs
import argparse
from saveevents import addevents
from mne.viz import plot_events


parser = argparse.ArgumentParser(description='Add events')

parser.add_argument('-i', type=str,help='Path to openbci txt file')
parser.add_argument('--control', type=str,required=True,help='Path to recordcontrol file')

                    
args = parser.parse_args()

openbcifile = args.i
controlfile = args.control

events,_,merged_events = addevents(controlfile)

Df = pd.read_csv(openbcifile,header=4)

chnames = ['C3','Cz','C4','EOG','A1','A2','TRIG']
chtypes = ['eeg','eeg','eeg','eeg','eeg','eeg','misc']

info = create_info(ch_names=chnames,sfreq=250,ch_types=chtypes)


## final data 
data = Df[[' EXG Channel 0', ' EXG Channel 1', ' EXG Channel 2',
       ' EXG Channel 3', ' EXG Channel 4', ' EXG Channel 5',' EXG Channel 7']].to_numpy().T

# MNE structure
raw =RawArray(data, info)
raw = raw.notch_filter(np.arange(50, 150, 50),picks=['eeg','misc'])
raw = raw.filter(l_freq=0.1, h_freq=40)

### reconstructing triggers
trigger = raw.get_data()[6,:]

triggernorm = (trigger - trigger.mean())/ trigger.std()
print(trigger.shape)

peaks,_ = find_peaks(triggernorm,height=[2,30],distance = 100) ### this will change if we change the way the trigger is generated

plt.plot(triggernorm)
plt.stem(peaks,10*np.ones_like(peaks),'--r')
plt.show()


events = np.vstack([peaks,np.zeros_like(peaks),np.ones_like(peaks)]).T
print(events)

## Rereferencing


fig = plot_events(events, sfreq=raw.info['sfreq'],
                            first_samp=raw.first_samp)


raw = raw.set_eeg_reference(ref_channels=['A1','A2'])

## Find EOG events 

#eog_events = find_eog_events(raw)

#epochs = Epochs(raw, events, tmin=-0.2, tmax=0.5, baseline=(-0.2,0),reject_by_annotation=True,
#                    preload=True)
raw.plot(events=events,block=True,scalings ={'eeg':'auto','misc':1e4})

signalfile_withevents = openbcifile[:-4] + '.fif'

raw.save(fname=signalfile_withevents,overwrite=True)

eventsfile = openbcifile[:-4] + '_eve.fif'

write_events(eventsfile,events,overwrite=True)


