# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:34:51 2024

@author: Edwin
"""

import numpy as np
from scipy import optimize
from hillmodel import ActState, Fce2Vce, ForceEQ, ForceLength, LEE2Force

#%% 
def GetStartValues(fsee0,gamma0,muspar):
    # Unravel parameter values
    w = muspar['w']
    lce_opt = muspar['lce_opt']
    kpee, ksee = muspar['kpee'], muspar['ksee']
    lpee0, lsee0 = muspar['lpee0'], muspar['lsee0']
    fmax = muspar['fmax']
    
    # Compute stuff..
    lcerel = np.linspace(1,1+w,1000)
    lce     = lcerel*lce_opt
    fce     = ForceLength(lcerel,muspar)[0]*fmax*ActState(1,lcerel,muspar)[0]
    lpee = lce
    epee = lpee-lpee0
    fpee = (epee<0)*0 + (epee>=0)*(kpee*epee**2) # [N] PEE force
    fsee = fpee+fce
    lsee = (fsee/ksee)**(1/2) + lsee0
    lmtc = lce+lsee
    
    dfseedlce = np.gradient(fsee,lce)
    iRelevant  = np.where(dfseedlce<0)[0]
    lcerel = lcerel[iRelevant]
    fsee = fsee[iRelevant]
    
    idx = np.argmin(np.abs(fsee-fsee0))
    
    lmtcGuess = lmtc[idx]
    bounds = ((lmtc[iRelevant[0]],lmtc[iRelevant[-1]]),)
    fun = lambda x: getVce(x,1,fsee0,muspar)**2
    lmtc0 = optimize.minimize(fun,lmtcGuess, bounds = bounds).x[0]
    lcerel0 = ForceEQ(lmtc0,gamma0,muspar)[1]
    
    return lmtc0, lcerel0
    
def getVce(lmtc,gamma,fsee,muspar):
    """
    getVce Computes CE velocity based on mtc length, gamma and see force.
    
    Inputs:
        lmtc        =   lmtc length [m]
        gamma       =   normalised concentration Ca2+ between the filaments
        fsee        =   SEE force [N]
        muspar      =   dict with muscle parameter values
    
    Outputs:
        vce         =   time-dertivative of CE length [m/s]
    """
    
    lsee    = (fsee/muspar['ksee'])**(1/2) + muspar['lsee0']
    lce     = lmtc-lsee
    lcerel  = lce/muspar['lce_opt']
    
    fce = LEE2Force(lmtc,lcerel,muspar)[2]
    q = ActState(gamma,lcerel,muspar)[0]
    vce = Fce2Vce(fce,q,lcerel,muspar)[0]
        
    # Output
    return vce

def SimuIso(t,state,inputs,muspar):
    """
    SimuIso Computes state-derivatives (gammad and vcerel) for a simulated
        isotonic experiment. That means, an experiment that starts with 
        length-control (to build up isometric force) and then followed
        by a force-controlled part. 
    
    Inputs:
        t           =   time [s]
        state       =   state of the muscle model (gamma: the normalised
                            concentration Ca2+ between the filament and 
                            lcerel: relative CE length, i.e. CE length divided 
                            by optimum CE length) [ , ]
        muspar      =   dict with muscle parameter values
    
    Outputs:
        gammad      =   time-derivative of gamma [1/s]
        vcerel      =   time-dertivative of relative CE length [1/s]
        y           =   list with other variables (i.e., lmtc, stim, q, etc.)
    """
    
    # Unravel parms & state
    gamma = state[0] # [ ]
    lcerel = state[1] # [ ] 
    
    tIso = inputs['tIso']
    stim = 1
    
    # Activation dynamics
    gamma_0 = muspar['gamma_0']
    gammad = (stim>=gamma)*((stim*(1-gamma_0)-gamma + gamma_0)/muspar['tact']) + (stim<gamma)*((stim*(1-gamma_0)-gamma + gamma_0)/muspar['tdeact']) # [1/s]
    q           = ActState(gamma,lcerel,muspar)[0]   
    
    # TMP!!!!!!!!!!!!!!!
    tStep = 10e-3
    fseeStep0 = 13.2
    dfseeStep = inputs['fseeDrop']-fseeStep0
    fseeStepDot = dfseeStep/tStep
    
    
    fseeControlled = (t>=tIso)*(t<=tIso+tStep)*(fseeStep0+(t-tIso)*fseeStepDot) + (t>tIso+tStep)*inputs['fseeDrop']
    
    lmtc = np.where(t < tIso, # switch between length and force control mode
                inputs['lmtc0'], # for length control mode
                lcerel * muspar['lce_opt'] + (fseeControlled / muspar['ksee'])**0.5 + muspar['lsee0']) # for force control mode
    # Check if we've exceeded shortening distance..
    lmtc0 = inputs['lmtc0']
    shorteningDistance = inputs['shorteningDistance']
    d = (lmtc-lmtc0)-shorteningDistance
    lmtc = np.where(d < 0, # switch between length and force control mode
                lmtc0+shorteningDistance, # for length control mode
                lmtc) # for force control mode
    
    lce = lcerel*muspar['lce_opt']
    lpee = lce
    lsee = lmtc-lce
    fsee,fpee,*_ = LEE2Force(lsee, lpee, muspar) # [N, N]
    fce = fsee-fpee
    vce,vcerel = Fce2Vce(fce,q,lcerel,muspar)[0:2] # [1/s]
    Hdot = -vce*muspar['a']+10

    # Outputs
    y = [lmtc, stim, fsee, Hdot]
    statedot = [gammad, vcerel, Hdot]
    return statedot, y

def endSimu(t,state,inputs,muspar):
    tIso = inputs['tIso']
    shorteningDistance = inputs['shorteningDistance']
    lmtc0 = inputs['lmtc0']
    
    # Unravel parms & state
    lcerel = state[1] # [ ] 
    
    # MTC dynamics
    if t<tIso: # we are in length control mode
        lmtc = lmtc0
        fsee,fpee,fce,fcerel = LEE2Force(lmtc, lcerel, muspar)[0:4] # [N, N, N, []]
    else: # we are in force control mode
        fsee = inputs['fseeDrop']
        lsee = (fsee/muspar['ksee'])**(1/2) + muspar['lsee0']
        lce = lcerel*muspar['lce_opt']
        lmtc = lce+lsee
    
    d = (lmtc-lmtc0)-shorteningDistance
    # print(d)
    return d