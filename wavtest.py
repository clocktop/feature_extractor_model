import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import bt_ic_model
import pickle

plt.style.use('./new_plot_style.mplstyle')

file_to_read = open("./bandpass_filters_tf.pkl","rb")

bpfs = pickle.load(file_to_read)

file_to_read = open("./fwr_tf_regression.pkl","rb")

FWR_model = pickle.load(file_to_read)

samplerate, data = wavfile.read("./10M_219.WAV")
time = np.linspace(0,len(data)/samplerate,len(data))
data_scaled = data[:,1]*(3.3/4095)

data_scaled = data_scaled - np.mean(data_scaled)

data_upsampled = signal.resample(data[:,1],len(time)*250)


time = np.linspace(0,len(data_upsampled)/samplerate,len(data_upsampled))
dt = 1/samplerate

pulse_trains, time_series = bt_ic_model.SimSignal(voltage_sig=data_upsampled,time_step=dt,bpfs=bpfs,FWR_model=FWR_model)

fig, axs = plt.subplots(9,1, figsize=[20,20])

for index, pulses_train in enumerate(pulse_trains):
    axs[index].step(time_series, pulse_trains[index], linewidth=0.75)

axs[8].plot(time_series,data_upsampled)

fig.savefig("upsampled_patch_data.png")

edms = np.zeros((len(pulse_trains),len(pulse_trains[0])))
for index, pulse_train in enumerate(pulse_trains):
    edm = bt_ic_model.EDM(pulse_train=pulse_train,alpha=2)
    edms[index,:] = edm

fig,ax = plt.subplots(figsize=[10,5])
im = ax.imshow(edms,aspect='auto',interpolation='none',vmin=0,cmap="bone")
ax.spines[:].set_visible(False)
#ax.set_xticks(np.arange(edms.shape[1]+1)-.5, minor=True)
ax.grid(False)
ax.set_yticks(np.arange(edms.shape[0]+1)-.5, minor=True)
ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
ax.tick_params(which="minor", bottom=False, left=False)
ax.set_ylabel("Feature Vector")
fig.savefig("upsampled_edm_feature_space.png")

fig, axs = plt.subplots(9,1, figsize=[20,20])

for index, pulses_train in enumerate(edms):
    axs[index].plot(time_series, edms[index], linewidth=0.2)

axs[8].plot(time_series,data_scaled)

fig.savefig("upsampled_edms.png")
