import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from scipy import signal
import matplotlib.pyplot as plt
import multiprocessing as mp
import threading 

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

def FWR_IAR(voltage_signal, model, dt, rate_crtl, pulse_train):
    current_sig = FWR(voltage_sig=voltage_signal,model=model)
    pulse_train[0],_ = IAF(current_sig=current_sig,dt=dt,rate_crtl=rate_crtl)
    return pulse_train

def BPF(voltage_sig, time_series, bpfs):
    filtered_sigs = [] 
    for tf in bpfs:
        temp = np.zeros(len(voltage_sig))
        _,temp,_ = signal.lsim((tf.num, tf.den), voltage_sig,time_series)
        filtered_sigs.append(temp)
    
    return filtered_sigs 

def SimSignal(voltage_sig, time_step, bpfs, FWR_model, rate_ctrl):
    endtime = len(voltage_sig)*time_step
    time_series = np.linspace(0,endtime,int(endtime/time_step))

    filtered_sigs = BPF(voltage_sig=voltage_sig, time_series=time_series,bpfs=bpfs)

    pulse_train_0 = [None] * 1
    pulse_train_1 = [None] * 1
    pulse_train_2 = [None] * 1
    pulse_train_3 = [None] * 1
    pulse_train_4 = [None] * 1
    pulse_train_5 = [None] * 1
    pulse_train_6 = [None] * 1
    pulse_train_7 = [None] * 1
    t0 = threading.Thread(target=FWR_IAR, args=(filtered_sigs[0], FWR_model, time_step, rate_ctrl,pulse_train_0))
    t1 = threading.Thread(target=FWR_IAR, args=(filtered_sigs[1], FWR_model, time_step, rate_ctrl,pulse_train_1))
    t2 = threading.Thread(target=FWR_IAR, args=(filtered_sigs[2], FWR_model, time_step, rate_ctrl,pulse_train_2))
    t3 = threading.Thread(target=FWR_IAR, args=(filtered_sigs[3], FWR_model, time_step, rate_ctrl,pulse_train_3))
    t4 = threading.Thread(target=FWR_IAR, args=(filtered_sigs[4], FWR_model, time_step, rate_ctrl,pulse_train_4))
    t5 = threading.Thread(target=FWR_IAR, args=(filtered_sigs[5], FWR_model, time_step, rate_ctrl,pulse_train_5))
    t6 = threading.Thread(target=FWR_IAR, args=(filtered_sigs[6], FWR_model, time_step, rate_ctrl,pulse_train_6))
    t7 = threading.Thread(target=FWR_IAR, args=(filtered_sigs[7], FWR_model, time_step, rate_ctrl,pulse_train_7))
    

    t0.start()
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
    t7.start()

    t0.join()
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
    t6.join()
    t7.join()

    #pool = mp.Pool(mp.cpu_count())

    #pulse_trains = [pool.apply(FWR_IAR, args=(filtered_sig, FWR_model, time_step, rate_ctrl)) for filtered_sig in filtered_sigs]
    
    #pool.close()
    
    #for filtered_sig in filtered_sigs:
    #    fwr_sig = FWR(voltage_sig=filtered_sig, model=FWR_model)
    #    pulse_train,_ = IAF(current_sig=fwr_sig, dt=time_step, rate_crtl=1)
    #    pulse_trains.append(pulse_train)
    pulse_trains = [pulse_train_0,pulse_train_1, pulse_train_2, pulse_train_3, pulse_train_4, pulse_train_5, pulse_train_6, pulse_train_7]
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