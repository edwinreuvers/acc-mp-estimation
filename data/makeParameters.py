# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:52:07 2022

@author: Edwin
"""

import pickle, os, sys
import numpy as np

cwd = os.path.dirname(os.path.abspath(__file__))
baseDir = os.path.join(cwd,'..')
dataDir = os.path.join(baseDir,'data')

for mus in ['GMs1','GMs2','GMs3']:
    muspar = {}
    
    #%% General
    # Force-length relationship
    muspar['w'] = 0.5;
    muspar['n'] = 2
    
    # Force-velocity relationship
    muspar['fasymp'] = 1.5
    muspar['slopfac'] = 2
    muspar['vfactmin'] = 0.1
    
    # Activation dynamics    
    muspar['a_act'] = -7.369163055099003
    muspar['b_act'] = np.array([5.170927028993413, 0.5955111970420514, 0])
    muspar['q0'] = 0.005
    muspar['gamma_0'] = 1e-5
    muspar['kCa'] = 0.8e-5
    muspar['tact'] = 1/37
    muspar['tdeact'] = 1/37
      
    fileName = os.path.join(dataDir,'GM_muspar.pkl')
    pickle.dump(muspar, open(fileName, 'wb'))
    
    #%% Rat specific
    if mus == 'GMs1':
        muspar['lce_opt'] = 1.32e-2 # [m]
        muspar['fmax'] = 13.39 # [N]
        muspar['ksee'] = 4.22e6
        muspar['kpee'] = 2.13e5
        muspar['lsee0'] = 2.83e-2 # [m]
        muspar['lpee0'] = 1.39e-2 # [m]
        # muspar['eseerelmax'] = (muspar['fmax']/4.22e6)**(1/2)/muspar['lsee0']
        # muspar['epeerelmax'] = (muspar['fmax']/2.13e5)**(1/2)/muspar['lpee0']
        muspar['a'] = 0.20*muspar['fmax']
        muspar['b'] = 3.15*muspar['lce_opt']
    elif mus == 'GMs2':
        muspar['lce_opt'] = 1.23e-2 # [m]
        muspar['fmax'] = 13.81 # [N]
        muspar['ksee'] = 3.64e6
        muspar['kpee'] = 1.65e5
        muspar['lsee0'] = 3.03e-2 # [m]
        muspar['lpee0'] = 1.32e-2 # [m]
        # muspar['eseerelmax'] = (muspar['fmax']/3.64e6)**(1/2)/muspar['lsee0']
        # muspar['epeerelmax'] = (muspar['fmax']/1.65e5)**(1/2)/muspar['lpee0']
        muspar['a'] = 0.13*muspar['fmax']
        muspar['b'] = 2.02*muspar['lce_opt']
    elif mus == 'GMs3':
        muspar['lce_opt'] = 1.12e-2 # [m]
        muspar['fmax'] = 12.28 # [N]
        muspar['ksee'] = 3.47e6
        muspar['kpee'] = 5.11e5
        muspar['lsee0'] = 2.65e-2 # [m]
        muspar['lpee0'] = 1.34e-2 # [m]
        # muspar['eseerelmax'] = (muspar['fmax']/3.47e6)**(1/2)/muspar['lsee0']
        # muspar['epeerelmax'] = (muspar['fmax']/5.11e5)**(1/2)/muspar['lpee0']
        muspar['a'] = 0.21*muspar['fmax']
        muspar['b'] = 3.73*muspar['lce_opt']
    else:
        print('Incorrect muscle selected!')
    
    filepath = os.path.join(dataDir,mus,'parameters',mus+'_OR.pkl')
    pickle.dump(muspar, open(filepath, 'wb'))