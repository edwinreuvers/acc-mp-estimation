# -*- coding: utf-8 -*-
"""
@author: edhmr, 2021
"""

import numpy as np
import hillmodel

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def qr(lmtc0,muspar,optsQR):
    
    #%% Motion parameters
    durIso  = optsQR['durIso']  # [s] duration of 1st and 2nd isometric phase
    durStep = optsQR['durStep'] # [s] duration of step
    dLmtc   = optsQR['dLmtc']   # [m] MTC length change during step
    tStim   = optsQR['tStim']   # [s] stim onset and offset time
    
    #%%
    # Create time-axis
    fs = 2000 # [Hz]
    time = np.arange(0,np.sum(durIso)+durStep,1/fs)
    
    ##% Get lmtc(t) & stim(t)
    lmtc = qr_lmtc(time,lmtc0,dLmtc,durIso,durStep)
    stim = make_stim(time,tStim[0],tStim[1])
    
    ##% Simlation to obtain fsee(t)
    gamma0 = muspar['gamma_0']
    lcerel0 = hillmodel.ForceEQ(lmtc[0],gamma0,muspar)[1]
    c_in = {}
    c_in['time'] = time
    c_in['lmtc'] = lmtc
    c_in['tStim']= tStim
    
    solstr = hillmodel.SolveSimuMTC(gamma0,lcerel0,muspar,c_in)[1]
    fsee = solstr[9]
    lcerel = solstr[4]
    gamma = solstr[3]
    fpee = solstr[10]
    fce = solstr[11]
       
    ##% Store
    data = np.vstack((time,lmtc,stim,fsee)).T
    
    #%% Outputs
    return data

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def sr(lmtc0,dLmtc,vRamp,muspar,optsSR):

    #%% Motion parameters
    durIso  = optsSR['durIso']  # [s] duration of 1st and 2nd isometric phase
    durStep = optsSR['durStep'] # [s] duration of step
    durRamp = optsSR['durRamp'] # [m] duration of ramp
    tStim   = optsSR['tStim']   # [s] stim onset and offset time
        
    # Create time-axis
    fs = 2000 # [Hz]
    time = np.arange(0,durIso+durStep+durRamp,1/fs)
    
    # Get lmtc(t) & stim(t)
    lmtc = sr_lmtc(time,lmtc0,dLmtc,vRamp,durIso,durStep,durRamp)
    stim = make_stim(time,tStim[0],tStim[1])
    
    # Simlation to obtain fsee(t)
    gamma0 = muspar['gamma_0']
    lcerel0 = hillmodel.ForceEQ(lmtc[0],gamma0,muspar)[1]
    c_in = {}
    c_in['time'] = time
    c_in['lmtc'] = lmtc
    c_in['tStim'] = tStim
    
    solstr = hillmodel.SolveSimuMTC(gamma0,lcerel0,muspar,c_in)[1]
    fsee = solstr[9]
    lcerel = solstr[4]
    gamma = solstr[3]
        
    ##% Save
    data = np.vstack((time,lmtc,stim,fsee)).T
    
    #%% Outputs
    return data

#%%
def isom(lmtc0,durStim,muspar):
    #%% Motion parameters     
    tStim = np.array([0.1, 0.1+durStim]) # [s] stim onset and offset time
    
    # Create time-axis
    fs = 2000 # [Hz]
    time = np.arange(0,0.3+durStim,1/fs)
    
    # Get lmtc(t) & stim(t)
    lmtc = lmtc0*np.ones(len(time))
    stim = make_stim(time,tStim[0],tStim[1])
    
    # Simlation to obtain fsee(t)
    gamma0 = muspar['gamma_0']
    lcerel0 = hillmodel.ForceEQ(lmtc[0],gamma0,muspar)[1]
    solmat = {}
    solmat['time'] = time
    solmat['lmtc'] = lmtc
    solmat['tStim'] = tStim
    
    solstr = hillmodel.SolveSimuMTC(gamma0,lcerel0,muspar,solmat)[1]
    fsee = solstr[9]
    lcerel = solstr[4]
    gamma = solstr[3]
    q = solstr[5]
    # print(q.max())
    
    #%% Store
    data = np.vstack((time,lmtc,stim,fsee)).T
    
    #%% Outputs
    return data

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def qr_lmtc(time,lmtc0,dLmtc,durIso,durStep):
    """
    qr_lmtc makes lmtc(t) of a quick-release experiment.
    
    Input:
        time    = time-axis [s]
        lmtc0   = MTC length at t=0 [m]
        dLmtc   = change in MTC length during the step [m]
        durIso  = duration of isometric phases (at the start and end) [s] 
        durStep = duration of step [s]
    Output:
        lmtc    = MTC-length over time [m]
    """
        
    #%% Check inputs
    if len(durIso) == 1:
        durIso = [durIso, durIso]
    elif len(durIso) == 2:
        durIso = durIso
    else:
        breakpoint()
    
    # Pre-allocate lmtc
    lmtc = np.zeros(np.size(time))*np.nan
    
    # Calculate (MTC) velocity during the step
    vStep = dLmtc/durStep
    
    # Find indices corresponding to different parts of 'movement'
    iP1 = time<=durIso[0] # indices corresponding to 1st isometricp phase
    iP2 = (time>durIso[0]) & (time<=(durIso[0]+durStep)) # indices corresponding to step phase
    iP3 = time>durIso[0]+durStep # indices corresponding to 2nd isometric phase
    
    # Fill in lmtc
    lmtc[iP1] = lmtc0 
    lmtc[iP2] = lmtc0+vStep*(time[iP2]-durIso[0])
    lmtc[iP3] = lmtc0+dLmtc
    
    #%% Ouputs
    return lmtc

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def sr_lmtc(time,lmtc0,dLmtc,vRamp,durIso,durStep,durRamp):
    """
    sr_lmtc makes lmtc(t) of a step-ramp experiment.
    Input:
        time    = time-axis [s]
        lmtc0   = MTC length at t=0 [m]
        dLmtc   = change in MTC length during the step [m]
        vRamp   = (MTC) velocity during the ramp [m/s]
        durIso  = duration of isometric phases (at the start and end) [s] 
        durStep = duration of step [s]
        durRamp = duration of ramp [s]
    Output:
        lmtc    = MTC-length over time [m]
    """
        
    #%%
    # Pre-allocate lmtc
    lmtc = np.zeros(np.size(time))*np.nan
    
    # Calculate (MTC) velocity during the step
    vStep = dLmtc/durStep
    
    # Find indices corresponding to different parts of 'movement'
    iP1 = time<=durIso # indices corresponding to 1st isometricp phase
    iP2 = (time>durIso) & (time<=(durIso+durStep)) # indices corresponding to step phase
    iP3 = (time>durIso+durStep) & (time<=(durIso+durStep+durRamp)) # indices corresponding to ramp phase
    iP4 = time>durIso+durStep+durRamp # indices corresponding to 2nd isometric phase
    
    # Fill in lmtc
    lmtc[iP1] = lmtc0 
    lmtc[iP2] = lmtc0+vStep*(time[iP2]-durIso)
    lmtc[iP3] = lmtc0+vStep*(time[iP2][-1]-durIso)+vRamp*(time[iP3]-(durIso+durStep))
    lmtc[iP4] = lmtc0+dLmtc+vRamp*durRamp
    
    ##% Ouputs
    return lmtc

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def isom_lmtc(time,lmtc0):
    """
    isom_lmtc make lmtc(t) of an isometric experiment.
    Input:
        time    = time-axis [s]
        lmtc0   = MTC length at t=0 [m]
    Output:
        lmtc    = MTC-length over time [m]
    """
    
    # Import functions
    import numpy as np
    
    # Pre-allocate lmtc
    lmtc = np.ones(np.size(time))*lmtc0
        
    return lmtc

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def make_stim(time,tStimOn,tStimOff):
    """
    make_stim makes stim(t).
    
    Input:
        time    = time-axis [s]
        tStimOn = stimulation onset time [s]
        tStimOff= stimulation offset time [s]
    Output:
        stim    = stimulation over time [m]
    """
        
    # Pre-allocate lmtc
    stim = np.zeros(np.size(time))
    
    # Get stim
    iOn = (time>=tStimOn) & (time<tStimOff)
    
    stim[iOn] = 1
    
    #%% Outputs
    return stim

