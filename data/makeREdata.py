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
from FuncMotionData import createRAMPdata

#%% 
mus = 'GMs1'
parvrs = 'O'

#%%
if mus == 'GMs1':
    dLmtcRange = np.array([-0.15, -0.30, -0.45, -0.70, -0.95, -1.20, -1.35, -1.50, -1.75])*1e-3 # [m] MTC lengthchange during step
    vRampRange = np.array([-05.0, -11.0, -17.0, -28.0, -40.0, -55.0, -64.0, -76.0, -92.0])*1e-3 # [m/s] MTC velocity during ramp
elif mus == 'GMs2':
    dLmtcRange = np.array([-0.15, -0.30, -0.50, -0.75, -1.05, -1.25, -1.50, -1.75, -2.00])*1e-3 # [m] MTC lengthchange during step
    vRampRange = np.array([-04.0, -08.0, -13.0, -21.0, -33.0, -44.0, -57.0, -72.0, -90.0])*1e-3 # [m/s] MTC velocity during ramp
elif mus == 'GMs3':
    dLmtcRange = np.array([-0.15, -0.30, -0.45, -0.70, -0.95, -1.10, -1.30, -1.50, -1.75])*1e-3 # [m] MTC lengthchange during step
    vRampRange = np.array([-04.0, -10.0, -16.0, -26.0, -38.0, -48.0, -60.0, -75.0, -90.0])*1e-3 # [m/s] MTC velocity during ramp


#%% Generate QR data 
# Load muscle parameter values
parFile = os.path.join(dataDir,mus,'Parameters',mus+'_OR.pkl')
muspar = pickle.load(open(parFile, 'rb'))
lmtcOpt = np.sqrt(muspar['fmax']/muspar['ksee'])+muspar['lsee0']+muspar['lce_opt']
lmtc0 = lmtcOpt+0.5e-3 # [m] MTC-length at t=0

if parvrs == 'O':
    subfldr = parvrs
    dataDir = os.path.join(baseDir,'0_Data','Data',mus,subfldr,'QR','')
    
    vRamp = vRampRange[3]
    # Make & save data
    fig, axs = plt.subplots(1,3)
    data = createRAMPdata(lmtc0,vRamp,muspar)
    axs[1].plot(data[:,0],data[:,1]) # lmtc
    axs[2].plot(data[:,0],data[:,3]) # fsee
        

    
