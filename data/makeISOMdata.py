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
from FuncMotionData import createACTdata

plt.close('all')

#%% Readout muscle parameter values
if not 'mus' in locals():
    mus = 'GMs3'

parFile = os.path.join(dataDir,mus,'Parameters',mus+'_OR.pkl')
muspar = pickle.load(open(parFile, 'rb'))
eseeMax = (muspar['fmax']/muspar['ksee'])**(1/2) # [mm] SEE elongation @ fmax
lmtcOpt = eseeMax + muspar['lsee0']+muspar['lce_opt']

#%% Generate ACT data
if mus == 'GMs1':
    lmtcRange = np.array([43,41,39,37, 43,41,39,37, 43,41,39,37])*1e-3 # [m] isometric MTC length
elif mus == 'GMs2':
    lmtcRange = np.array([44,42,40,38, 44,42,40,38, 44,42,40,38])*1e-3 # [m] isometric MTC length
elif mus == 'GMs3':
    lmtcRange = np.array([39,37,35,33, 39,37,35,33, 39,37,35,33])*1e-3 # [m] isometric MTC length
# durStimRange = np.array([5,5,5, 10,10,10, 15,15,15])*1e-3 # [s] stimulation duration
durStimRange = np.array([35,35,35,35 , 65,65,65, 65, 95,95,95,95])*1e-3 # [s] stimulation duration

#%% Simulate ISOM data - 'Experimental data'
dataDirISOM = os.path.join(dataDir,mus,'dataExp','ISOM','')

# First delete all files in folder
files = glob.glob(dataDirISOM+r'\*')    
for iFile,filename in enumerate(files):
    os.remove(filename)

# Make & save data
plt.close('all')
fig, ax = plt.subplots(3,4)
for iFile,(lmtc0,durStim) in enumerate(zip(lmtcRange,durStimRange)):
    data = createACTdata(lmtc0,durStim,muspar)
    
    # Check
    iT = np.mod(iFile,4)
    ax[0,iT].plot(data[:,0],data[:,2]) # sitm
    ax[1,iT].plot(data[:,0],data[:,1]) # lmtc
    ax[2,iT].plot(data[:,0],data[:,3]) # fsee
    
    fileName = mus+'_ISOM'+'{:02d}'.format(iFile+1)+'_OR.csv'
    pd.DataFrame(data).to_csv(dataDirISOM+fileName,index=False, 
                                header=['time [s]','Lmtc [m]','STIM [ ]','Fsee [N]'])
    print('ISOM: '+mus+', '+'OR'+', File: '+str(iFile+1))

#%% Simulate ISOM data - Monte Carlo simulations
for iPar in range(1,51):
    dataDirISOM = os.path.join(dataDir,mus,'dataMC',f'{iPar:{"02d"}}','ISOM','')
    
    # Define MTC shifts, we will not choose a different LMTC length at the start, 
    # because it's unrealistic that one will change this for every SR in an experiment
    np.random.seed(iPar)
    shifts = np.random.rand(12)*1e-3
    
    # First delete all files in folder
    files = glob.glob(dataDirISOM+r'\*')    
    for iFile,filename in enumerate(files):
        os.remove(filename)
    
    # Make & save data
    plt.close('all')
    fig, ax = plt.subplots(3,4)
    for iFile,(lmtc0,durStim) in enumerate(zip(lmtcRange,durStimRange)):
        mcpar = muspar.copy()
        # Distrub data
        eseeMax = lmtcOpt+shifts[iFile]-muspar['lce_opt']-muspar['lsee0'] # [mm] SEE elongation @ fmax
        mcpar['ksee'] = mcpar['fmax']/(eseeMax)**2
        data = createACTdata(lmtc0,durStim,mcpar)
        
        # Check
        iT = np.mod(iFile,4)
        ax[0,iT].plot(data[:,0],data[:,2]) # sitm
        ax[1,iT].plot(data[:,0],data[:,1]) # lmtc
        ax[2,iT].plot(data[:,0],data[:,3]) # fsee
        
        fileName = mus+'_ISOM'+'{:02d}'.format(iFile+1)+'_MC{:02d}'.format(iPar)+'.csv'
        pd.DataFrame(data).to_csv(dataDirISOM+fileName,index=False, 
                                  header=['time [s]','Lmtc [m]','STIM [ ]','Fsee [N]'])
        print('ISOM: '+mus+', '+'MC{:02d}'.format(iPar)+', File: '+str(iFile+1))