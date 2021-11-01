import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from scipy import signal
import matplotlib.pyplot as plt

def IAF(current_sig, dt, rate_crtl):
    if rate_crtl:
        c = 500e-15
    else:
        c = 250e-15

    Vx = np.zeros(len(current_sig))
    pulse_train = np.zeros(len(current_sig))

    for index in range(len(current_sig)):
        if index == 0:
           pass
        else:
            Vx[index] = Vx[index - 1] + (1/c)*dt*current_sig[index]

            if Vx[index] > 0.6:
                Vx[index] = 0
                pulse_train[index] = 1
    
    return pulse_train,Vx

def FWR(voltage_sig, model):
    X = np.array(voltage_sig).reshape(-1,1)
    current_sig = model.predict(X)
    return current_sig*1e-9

def BPF(voltage_sig, time_series, bpfs):
    filtered_sigs = [] 
    for tf in bpfs:
        temp = np.zeros(len(voltage_sig))
        _,temp,_ = signal.lsim((tf.num, tf.den), voltage_sig,time_series)
        filtered_sigs.append(temp)
    
    return filtered_sigs 

def SimSignal(voltage_sig, time_step, bpfs, FWR_model):
    endtime = len(voltage_sig)*time_step
    time_series = np.linspace(0,endtime,int(endtime/time_step))

    filtered_sigs = BPF(voltage_sig=voltage_sig, time_series=time_series,bpfs=bpfs)

    pulse_trains = []

    for filtered_sig in filtered_sigs:
        fwr_sig = FWR(voltage_sig=filtered_sig, model=FWR_model)
        pulse_train,_ = IAF(current_sig=fwr_sig, dt=time_step, rate_crtl=0)
        pulse_trains.append(pulse_train)

    return pulse_trains, time_series

def EDM(pulse_train, alpha):
    y = np.zeros(len(pulse_train))
    y_prior = 0
    lamb = 1/(np.power(2, alpha))
    for index, x in enumerate(pulse_train):
        y[index] = y_prior - lamb*(y_prior - x)
        y_prior = y[index]
    
    return y

def VizPulseTrainDensity(data):
    fig,ax = plt.subplots(figsize=[10,5])
    im = ax.imshow(data,aspect='auto',interpolation='none',vmin=0,cmap="bone")
    ax.spines[:].set_visible(False)
    #ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.grid(False)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_ylabel("Feature Vector")
    return fig, ax