# -*- coding: utf-8 -*-
"""
Created on Tue May 24 17:13:47 2022

@author: Edwin
"""

import numpy as np
import pandas as pd
try:
    from hillmodel import ForceEQ, SolveSimuMTC
except:
    from .hillmodel import ForceEQ, SolveSimuMTC

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def simulate_file(filepath, muspar):
    """
    Load one experimental file, run the simulation, and return a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the experimental CSV file.
    muspar : dict
        Muscle parameter dictionary (from pickle).

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with time, Lmtc, STIM, and Fsee.
    """   
    
    # Read experimental data
    df_exp = pd.read_csv(filepath)
    data = df_exp.T.to_numpy()
    time, lmtc, stim = data[0:3]
    _, tStimOn, tStimOff = get_stim(time, stim)

    # Get initial states for simulation
    gamma0 = muspar['gamma_0']
    lcerel0 = ForceEQ(lmtc[0], gamma0, muspar)[1]

    # Inputs for simulation
    solmat = {
        'time': time,
        'lmtc': lmtc,
        'tStim': np.vstack((tStimOn, tStimOff)).T
    }
    
    # Run simulation
    solstr = SolveSimuMTC(gamma0, lcerel0, muspar, solmat)[1]
    fsee = solstr[9]
    
    # Store in dataframe
    df_sim = pd.DataFrame(
        np.vstack((time, lmtc, stim, fsee)).T,
        columns=['time [s]', 'Lmtc [m]', 'STIM [ ]', 'Fsee [N]']
    )

    return df_sim, df_exp

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_stim(time,signal):
    """
    Finds the start and stop times of stimulation pulse trains in a signal.

    Inputs:
    - signal: 1D array-like stimulation over time (e.g., list or numpy array)

    Outputs:
    - pulse_trains: List of tuples where each tuple contains (start, stop) indices of a pulse train
    """
    # Determine an appropriate threshold as halfway between min and max of the signal
    threshold = (np.max(signal) + np.min(signal)) / 2
    
    # Convert signal to binary form based on the threshold
    binary_signal = np.where(signal > threshold, 1, 0)
    
    # Find the start and stop indices of all pulses
    iStartPulse = np.where(np.diff(binary_signal, prepend=0) == 1)[0]
    iStopPulse = np.where(np.diff(binary_signal, prepend=0) == -1)[0]
    
    if len(iStartPulse) == 0 or len(iStopPulse) == 0:
        return []
    
    # Estimate the minimum gap between pulse trains by finding the most common pulse width
    PulseWidths = iStopPulse - iStartPulse
    PulseWidthMed = np.median(PulseWidths)
    
    # Separate the start and stop indices into different pulse trains
    iStartTrain = [int(iStartPulse[0])]  # Convert to Python int
    iStopTrain = []
        
    for i in range(1, len(iStartPulse)):
        # If the gap between current stop and next start is too large, close the current train
        if iStartPulse[i] - iStopPulse[i-1] > PulseWidthMed:
            iStopTrain.append(int(iStopPulse[i-1]))  # Convert to Python int
            iStartTrain.append(int(iStartPulse[i]))  # Convert to Python int
    
    # Add the final pulse train stop
    iStopTrain.append(int(iStopPulse[-1]))  # Convert to Python int
        
    stimModel = get_block_signal(signal,iStartTrain, iStopTrain)
    tStimOn = time[iStartTrain]
    tStimOff = time[iStopTrain]
    return stimModel, tStimOn, tStimOff

def get_block_signal(signal, pulse_train_starts, pulse_train_stops):
    """
    Creates a block-shaped signal around the pulse trains.
    
    Iputs:
    - signal: 1D array-like signal over time (e.g., list or numpy array)
    - pulse_train_starts: List of start indices of the pulse trains (Python integers)
    - pulse_train_stops: List of stop indices of the pulse trains (Python integers)
    
    Outputs:
    - block_signal: 1D numpy array of the same length as the input signal with block shapes (height 1) around pulse trains
    """
    # Initialize the block signal as a zero array with the same length as the input signal
    block_signal = np.zeros_like(signal)
    
    # Set the block signal to 1 between each pair of start and stop indices
    for start, stop in zip(pulse_train_starts, pulse_train_stops):
        block_signal[start:stop] = 1
    
    return block_signal