# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 08:38:09 2025

@author: Edwin
"""

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
from FuncGen import floor
from FuncMotionData import createQRdata

plt.close('all')

#%% Readout muscle parameter values
if not 'mus' in locals():
    mus = 'GMz2'

parFile = os.path.join(dataDir,mus,'Parameters',mus+'_OR.pkl')
muspar = pickle.load(open(parFile, 'rb'))
eseeMax = (muspar['fmax']/muspar['ksee'])**(1/2) # [mm] SEE elongation @ fmax
lmtcOpt = eseeMax + muspar['lsee0']+muspar['lce_opt']

#%% Simulate QR data - 'Experimental data'
# Setting which are similar for each QR
optsQR = {
    'durIso':   [0.3, 0.2], # [s] duration of 1st and 2nd isometric phase
    'durStep':  10e-3,      # [s] duration of step
    'dLmtc':    -0.2e-3,     # [m] MTC length change during step 
    'tStim':    [0.1, 0.4]  # [s] stim onset and offset time
    }
dataDirQR = os.path.join(dataDir,mus,'dataExp','QR','')

#%% Simulate QR data - Monte Carlo simulations
for iPar in range(1,51):
    dataDirQR = os.path.join(dataDir,mus,'dataMC',f'{iPar:{"02d"}}','QR','')
    
    # Define over which range we will simulate QR, this is a different range (i.e., slightly shifted). 
    # This is what one will (or should do) in experiment too!
    np.random.seed(iPar)
    shifts = np.random.rand(10)*1e-3
    lmtcOptShifted = lmtcOpt+shifts[0]
    lmtcRange = np.linspace(floor(lmtcOptShifted,3)+3e-3,floor(lmtcOptShifted,3)-6e-3,10)
    
    # # First delete all files in folder
    # files = glob.glob(dataDirQR+r'\*')    
    # for iFile,filename in enumerate(files):
    #     os.remove(filename)
    
    # Make & save data
    plt.close('all')
    fig, ax = plt.subplots(3,1)
    for iFile,lmtc0 in enumerate(lmtcRange[4:]):
        mcpar = muspar.copy()
        # Distrub data
        eseeMax = lmtcOpt+shifts[iFile]-muspar['lce_opt']-muspar['lsee0'] # [mm] SEE elongation @ fmax
        mcpar['ksee'] = mcpar['fmax']/(eseeMax)**2
        data = createQRdata(lmtc0,mcpar,optsQR)
        
        # Check
        ax[0].plot(data[:,0],data[:,2]) # sitm
        ax[1].plot(data[:,0],data[:,1]*1e3) # lmtc
        ax[2].plot(data[:,0],data[:,3]) # fsee
        print(data[:,3].max())
        sys.exit()
        fileName = mus+'_QR'+'{:02d}'.format(iFile+1)+'_MC{:02d}'.format(iPar)+'.csv'
        pd.DataFrame(data).to_csv(dataDirQR+fileName,index=False, 
                                  header=['time [s]','Lmtc [m]','STIM [ ]','Fsee [N]'])
        print('QR: '+mus+', '+'MC{:02d}'.format(iPar)+', File: '+str(iFile+1))
