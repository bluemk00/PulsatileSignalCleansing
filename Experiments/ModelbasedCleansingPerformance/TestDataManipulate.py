import numpy as np
import os

# Saturation to ABP maximum artifact
def SatABPmax(SatABPmaxSet):
    SatABPmaxEnd = np.random.randint(501, 1000)
    SliceStart = np.random.randint(0, SatABPmaxEnd - 500)
    SliceEnd = SliceStart + 500
    MinDias = np.min(SatABPmaxSet)
    MaxSyst = np.max(SatABPmaxSet)
    VecToSat = np.arange(SatABPmaxEnd)
    TmpMax = np.random.uniform(low=MaxSyst * 1.1, high=(MaxSyst + 1.0) * 0.5)
    VecToSat = np.tanh((np.pi * VecToSat * np.random.uniform(low=0.1, high=0.9)) / 100) * (TmpMax - MinDias) + MinDias
    t = np.linspace(0, 15, 500)
    exp_t = np.exp(-t)
    SatABPmaxSet[-500:] = exp_t * SatABPmaxSet[-500:] + (1 - exp_t) * VecToSat[SliceStart:SliceEnd]
    return SatABPmaxSet

# Saturation to ABP minimum artifact
def SatABPmin(SatABPminSet):
    SatABPminEnd = np.random.randint(501, 1000)
    SliceStart = np.random.randint(0, SatABPminEnd - 500)
    SliceEnd = SliceStart + 500
    MinDias = np.min(SatABPminSet) * np.random.uniform(low=0.7, high=0.99)
    MeanSyst = np.mean(SatABPminSet)
    LeftSize = int(SatABPminEnd * 0.8)
    RightSize = SatABPminEnd - LeftSize
    LeftVecToSat = np.arange(LeftSize)
    RightVecToSat = np.arange(RightSize)
    LeftVecToSat = (1 - np.tanh((np.pi * LeftVecToSat * np.random.uniform(low=0.1, high=0.3)) / 100)) * (MeanSyst * np.random.normal(loc=1.05, scale=0.05) - MinDias) + MinDias
    RightVecToSat = (np.tanh((np.pi * (RightVecToSat - RightSize) * np.random.uniform(low=0.7, high=0.99)) / 100) + 1) * (MeanSyst * np.random.normal(loc=0.95, scale=0.05) - MinDias) + np.min(LeftVecToSat)
    VecToSat = np.concatenate([LeftVecToSat, RightVecToSat])
    t = np.linspace(0, 10, 500)
    exp_t = np.exp(-t)
    SatABPminSet[-500:] = exp_t * SatABPminSet[-500:] + (1 - exp_t) * VecToSat[SliceStart:SliceEnd]
    return SatABPminSet

# Reduced pulse pressure artifact
def RedPP(RedABPSet):
    RedABP = RedABPSet[-500:]
    InitDias = np.min(RedABP)
    eta = np.random.uniform(0.1, 0.5)
    SaledRedABP = (RedABP - InitDias) * np.linspace(eta + np.random.uniform(0.1, 0.5), eta, len(RedABP)) + InitDias
    RedABPSet[-500:] = SaledRedABP
    return RedABPSet 

# Impulse artifact
def Impulse(ImpSet):
    SecSize = 100
    ImpSize = 5
    fs = np.random.uniform(100, 500)
    VecToSat = np.arange(-(SecSize // 2), (SecSize // 2)) + 0.01
    ImpulseArt = 0.02 * np.sin((np.pi * VecToSat) / (fs * np.random.uniform(0.1, 0.4, ImpSize)[:, None])) / (np.pi * VecToSat / fs)
    ImpulseArt = np.random.choice([-1, 1], size=ImpSize, replace=True)[:, None] * ImpulseArt
    ImpulseArt = np.reshape(ImpulseArt, (SecSize * ImpSize)).copy()
    ImpSet[-(SecSize * ImpSize):] += ImpulseArt
    ImpSet = np.clip(ImpSet, 0.01, 0.99)
    return ImpSet

# Example usage of adding artifacts
def add_artifacts(abp, artifact_func, filename):
    noised_abp = abp.copy()
    for i in range(len(abp)):
        noised_abp[i, -500:] = artifact_func(noised_abp[i, -500:])
    np.save(filename, noised_abp)

# Ensure directories exist
os.makedirs('./TestDataSet/ABP/Original', exist_ok=True)

# Load ABP data
abp = np.load('../../../../BioSignalCleaning/A.Data/A2.Processed/ABP/VitalTotalSet.npy')
abp = abp[:, :3000]
np.save('./TestDataSet/ABP/Original/VitalDB_ABP_high_qual.npy', abp)

# Apply different artifacts and save
add_artifacts(abp, SatABPmax, './TestDataSet/ABP/Original/VitalDB_ABP_satmax.npy')
add_artifacts(abp, SatABPmin, './TestDataSet/ABP/Original/VitalDB_ABP_satmin.npy')
add_artifacts(abp, RedPP, './TestDataSet/ABP/Original/VitalDB_ABP_reduced.npy')
add_artifacts(abp, Impulse, './TestDataSet/ABP/Original/VitalDB_ABP_impulse.npy')

# High frequency noise
noised_abp = abp.copy()
for i in range(len(abp)):
    noised_abp[i, -500:] += np.random.normal(loc=0.0, scale=0.05, size=noised_abp[i, -500:].shape)
noised_abp = np.clip(noised_abp, 0.0, 1.0)
np.save('./TestDataSet/ABP/Original/VitalDB_ABP_highfreq.npy', noised_abp)

# Incomplete noise
noised_abp = abp.copy()
for i in range(len(abp)):
    m = np.mean(noised_abp[i, -500:])
    noised_abp[i, -500:] = np.random.normal(loc=m, scale=0.01, size=noised_abp[i, -500:].shape)
noised_abp = np.clip(noised_abp, 0.0, 1.0)
np.save('./TestDataSet/ABP/Original/VitalDB_ABP_incomplete.npy', noised_abp)

# Create mask
mask = np.zeros_like(abp)
mask[:, -500:] = 1
np.save('./TestDataSet/ABP/Original/VitalDB_ABP_mask1.npy', mask)  # For HI-VAE and GP-VAE
np.save('./TestDataSet/ABP/Original/VitalDB_ABP_mask0.npy', 1 - mask)  # For BDC and SNM
