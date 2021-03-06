from mne.io import read_raw_edf,RawArray,read_raw_fif
from mne.export import export_raw
from mne import find_events,annotations_from_events,write_events,create_info,read_events, Epochs
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

parser.add_argument('-i', type=str,help='Path to openbci edf file generated by import_reref.py')

args = parser.parse_args()
edffile = args.i

eventsfile = edffile[:-4] + '_eve.fif'

raw = read_raw_fif(edffile,preload=True)
events = read_events(eventsfile)

raw.plot(events=events,block=True,scalings ={'eeg':'auto','misc':1e4})
raw = raw.filter(l_freq=0.1, h_freq=40)

epochs = Epochs(raw, events, tmin=-0.2, tmax=0.5,baseline=(-0.2,0),reject_by_annotation=True,
                    preload=True)
evoked = epochs.average()
evoked.plot(picks=['C3','C4','Cz'])