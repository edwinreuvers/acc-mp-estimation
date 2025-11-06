# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:20:18 2022

@author: Edwin
"""

#%% Load modules
import pickle, os, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cwd = os.path.dirname(os.path.abspath(__file__))
baseDir = os.path.join(cwd,'..')
dataDir = os.path.join(baseDir,'data','') 
funcDir = os.path.join(baseDir,'functions')
sys.path.append(funcDir)
from FuncMotionData import createSRdata
from FuncMus import Fce2Vce

plt.close('all')

#%% Readout muscle parameter values
if not 'mus' in locals():
    mus = 'GMs1'

parFile = os.path.join(dataDir,mus,'Parameters',mus+'_OR.pkl')
muspar = pickle.load(open(parFile, 'rb'))
eseeMax = (muspar['fmax']/muspar['ksee'])**(1/2) # [mm] SEE elongation @ fmax
lmtcOpt = eseeMax + muspar['lsee0']+muspar['lce_opt']
lmtc0 = lmtcOpt+0.5e-3 # [m] MTC-length at t=0

#%% Simulate SR data - 'Experimental data'
fceRange = np.array([0.85, 0.72, 0.60, 0.49, 0.38, 0.28, 0.20, 0.14, 0.09])*muspar['fmax']

# Setting which are similar for each SR
optsSR = {
    'durIso':   0.3,        # [s] duration of 1st isometric phase
    'durStep':  10e-3,      # [s] duration of step
    'durRamp':  0.2,        # [m] MTC length change during step
    'tStim':    [0.1, 0.4]  # [s] stim onset and offset time
    }
dataDirSR = os.path.join(dataDir,mus,'dataExp','SR','')

# First delete all files in folder
files = glob.glob(dataDirSR+r'\*')    
for iFile,filename in enumerate(files):
    os.remove(filename)

# Make & save data
plt.close('all')
fig, ax = plt.subplots(3,1)
for iFile,fce in enumerate(fceRange):
    vRamp = Fce2Vce(fce,1,1,muspar)[0]
    esee = (fceRange[iFile]/muspar['ksee'])**0.5
    eseeMax = (muspar['fmax']/muspar['ksee'])**0.5
    dLmtc = np.round((esee-eseeMax)+(vRamp*1e-2)*0.5,5)
    data = createSRdata(lmtc0,dLmtc,vRamp,muspar,optsSR)
    
    # Check
    ax[0].plot(data[:,0],data[:,2]) # sitm
    ax[1].plot(data[:,0],data[:,1]) # lmtc
    ax[2].plot(data[:,0],data[:,3]) # fsee
    fileName = mus+'_SR'+'{:02d}'.format(iFile+1)+'_OR.csv'
    pd.DataFrame(data).to_csv(dataDirSR+fileName,index=False, 
                              header=['time [s]','Lmtc [m]','STIM [ ]','Fsee [N]'])
    print('SR: '+mus+', '+'OR'+', File: '+str(iFile+1))

#%% Simulate SR data - Monte Carlo simulations
for iPar in range(1,51):
    dataDirSR = os.path.join(dataDir,mus,'dataMC',f'{iPar:{"02d"}}','SR','')
    
    # Define MTC shifts, we will not choose a different LMTC length at the start, 
    # because it's unrealistic that one will change this for every SR in an experiment
    np.random.seed(iPar)
    shifts = np.random.rand(9)*1e-3
    
    # First delete all files in folder
    files = glob.glob(dataDirSR+r'\*')    
    for iFile,filename in enumerate(files):
        os.remove(filename)
    
    # Make & save data
    plt.close('all')
    fig, ax = plt.subplots(3,1)
    for iFile,fce in enumerate(fceRange):
        mcpar = muspar.copy()
        # Distrub data
        eseeMax = lmtcOpt+shifts[iFile]-muspar['lce_opt']-muspar['lsee0'] # [mm] SEE elongation @ fmax
        mcpar['ksee'] = mcpar['fmax']/(eseeMax)**2
    
        vRamp = Fce2Vce(fceRange[iFile],1,1,mcpar)[0]
        esee = (fceRange[iFile]/mcpar['ksee'])**0.5
        eseeMax = (mcpar['fmax']/mcpar['ksee'])**0.5
        dLmtc = np.round((esee-eseeMax)+(vRamp*1e-2)*0.5,5)
        data = createSRdata(lmtc0,dLmtc,vRamp,mcpar,optsSR)
        
        # Check
        ax[0].plot(data[:,0],data[:,2]) # sitm
        ax[1].plot(data[:,0],data[:,1]*1e3) # lmtc
        ax[2].plot(data[:,0],data[:,3]) # fsee
        
        fileName = mus+'_SR'+'{:02d}'.format(iFile+1)+'_MC{:02d}'.format(iPar)+'.csv'
        pd.DataFrame(data).to_csv(dataDirSR+fileName,index=False, 
                                  header=['time [s]','Lmtc [m]','STIM [ ]','Fsee [N]'])
        print('SR: '+mus+', '+'MC{:02d}'.format(iPar)+', File: '+str(iFile+1))