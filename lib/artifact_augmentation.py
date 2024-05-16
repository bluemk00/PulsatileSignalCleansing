import random
import numpy as np
import math
from scipy import signal

import tensorflow as tf
from tensorflow.keras import backend as K, losses


class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self, filename):
        self.filename = filename

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        with open(self.filename, 'w') as f:
            f.write("Epoch,Loss,Validation Loss\n")

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        with open(self.filename, 'a') as f:
            f.write(f"{epoch},{logs.get('loss')},{logs.get('val_loss')}\n")


class DataBatch(tf.keras.utils.Sequence):
    
    def __init__(self, DataInp, BatchSize, shuffle = False, outptype = 0, *args, **kwargs):
        
        self.DataInp = DataInp
        self.DataLen = DataInp.shape[0]
        self.TimeLen = DataInp.shape[1]

        self.BatchSize = BatchSize
                
        self.Indices = np.arange(self.DataLen)
        self.shuffle = shuffle
        self.outptype = outptype
        '''
        outptype
        0 : (OutpData, OutpData_diff)
        1 : (OutpData, OutAmplitude)
        2 : (OutpData, OutpData_diff, OutAmplitude)
        3 : (OutpData, shtr)
        4 : (OutpData, OutpData_diff, shtr)
        '''
        
    def __len__(self):
        # returns the number of batches
        return math.ceil(self.DataLen / self.BatchSize)

    
    def __getitem__(self, idx):
        # returns one batch
        BatchIndex = self.Indices[idx*self.BatchSize:(idx+1)*self.BatchSize]
        SelectedBatch = self.DataInp[BatchIndex]
        
        # copying clean data as output
        OutpData = SelectedBatch[:,-500:].copy()
        OutpData_diff = OutpData[:,1:] - OutpData[:,:-1]
        
        # generating local & global noise sampling index
        WNR = 0.85 # whole noise rate
        NoiseRate = [WNR*0.23, WNR*0.23, WNR*0.18, WNR*0.18, WNR*0.18]
        NoiseSize = [int(len(BatchIndex)*NR) for NR in NoiseRate]
        NormalSize = len(BatchIndex) - sum(NoiseSize)
        NoiseBatchInd = np.concatenate([np.ones(i)*num for num,i in enumerate([NormalSize]+NoiseSize)]) # 0 : normal
        NoiseBatchInd = np.array([int(ind+10) if num%4==1 else int(ind+20) if num%4==2 else int(ind+30) if num%4==3 else int(ind) for num, ind in enumerate(NoiseBatchInd)])
        NoiseBatchInd = np.random.permutation(NoiseBatchInd)

        # select for saturation to ABP maximum
        SelectedBatch[NoiseBatchInd%10==1] = SatABPmax(SelectedBatch[NoiseBatchInd%10==1])

        # select for saturation to ABP minmum
        SelectedBatch[NoiseBatchInd%10==2] = SatABPmin(SelectedBatch[NoiseBatchInd%10==2])

        # Reduced pulse pressure artifact
        SelectedBatch[NoiseBatchInd%10==3] = RPP(SelectedBatch[NoiseBatchInd%10==3])

        # High frequency artifact
        SelectedBatch[NoiseBatchInd%10==4] = IPP(SelectedBatch[NoiseBatchInd%10==4])

        # Impulse artifact
        SelectedBatch[NoiseBatchInd%10==5] = ImpNoise(SelectedBatch[NoiseBatchInd%10==5])
        
        # Global Low frequency variance (Assume that LF rate is 0.5.)
        SelectedBatch[NoiseBatchInd<10] = LFVar(SelectedBatch[NoiseBatchInd<10])
        mask = (NoiseBatchInd>=10) & (NoiseBatchInd<20)
        SelectedBatch[mask] = HFNoise(SelectedBatch[mask])
        SelectedBatch[NoiseBatchInd>29] = TimeLagVar(SelectedBatch[NoiseBatchInd>29])
        
        InpData = SelectedBatch.copy()

        ## FFT Oup
        if (self.outptype == 1) or (self.outptype == 2):
            OutCasted = tf.cast(OutpData, tf.complex64) 
            Outfft = tf.signal.fft(OutCasted)[:, :(OutCasted.shape[-1]//2+1)]
            OutAmplitude = tf.abs(Outfft[:,1:])*(1/(Outfft.shape[-1])) # ?ÅÎ???ÏßÑÌè≠ Relative amplitude

        ## HBT
        if (self.outptype == 3) or (self.outptype == 4):
            N = OutpData.shape[-1]
            sftr = tf.signal.fft(tf.cast(OutpData, tf.complex64) )
            sffr = np.fft.fftfreq(n=N, d=1/N)
            sfhtr = (-1) * tf.cast(tf.complex(0.0,1.0), tf.complex64)[None,None] * tf.cast(tf.math.sign(sffr), tf.complex64)[None] * sftr
            shtr = tf.math.real(tf.signal.ifft(sfhtr))
            # OutHBT = tf.complex(tf.cast(OutpData, tf.float32), tf.cast(shtr, tf.float32))
        
        if self.outptype == 0:
            return (InpData), (OutpData, OutpData_diff)
        elif self.outptype == 1:
            return (InpData), (OutpData, OutAmplitude)
        elif self.outptype == 2:
            return (InpData), (OutpData, OutpData_diff, OutAmplitude)
        elif self.outptype == 3:
            return (InpData), (OutpData, shtr)
        elif self.outptype == 4:
            return (InpData), (OutpData, OutpData_diff, shtr)

    
        
    def on_epoch_end(self,):
        self.Indices = np.arange(self.DataLen)
        if self.shuffle:
            np.random.shuffle(self.Indices)
            
            
            
## saturation to ABP maximum artifact
def SatABPmax(SatABPmaxSet):

    SatABPmaxEnd = np.random.randint(501, 1000)

    SliceStart = np.random.randint(0, SatABPmaxEnd-500)
    SliceEnd = SliceStart + 500

    MinDias = np.min(SatABPmaxSet[:, -1000:-500], axis=-1, keepdims=True)
    VecToSat = np.arange(SatABPmaxEnd)
    TmpMax = np.random.uniform(low=0.7, high=0.95, size=(1,))
    VecToSat = np.tanh((np.pi * VecToSat * np.random.uniform(low=0.1, high=0.9, size=(1,))) / 100) * (TmpMax - MinDias) + MinDias

    SatABPmaxSet[:, -500:] = VecToSat[:, SliceStart:SliceEnd]

    return SatABPmaxSet


## saturation to ABP minimum artifact
def SatABPmin(SatABPminSet):

    SatABPminEnd= np.random.randint(501, 1000)

    SliceStart = np.random.randint(0, SatABPminEnd - 500)
    SliceEnd = SliceStart + 500     

    MinDias = np.min(SatABPminSet[:, -1000:-500], axis=-1, keepdims=True) * np.random.uniform(low=0.7, high=0.99, size=(1,))
    MeanSyst = np.mean(SatABPminSet[:, -600:-500], axis=-1, keepdims=True)

    LeftSize= int(SatABPminEnd * 0.8)
    RightSize = SatABPminEnd - LeftSize
    LeftVecToSat = np.arange(LeftSize)
    RightVecToSat = np.arange(RightSize)

    LeftVecToSat = (1 - np.tanh((np.pi * LeftVecToSat * np.random.uniform(low=0.1, high=0.3, size=(1,))) / 100)) * (MeanSyst * np.random.normal(loc=1.05, scale=0.05, size=(1,)) - MinDias) + MinDias
    RightVecToSat = (np.tanh((np.pi * (RightVecToSat - RightSize) * np.random.uniform(low=0.7, high=0.99, size=(1,))) / 100) + 1) * (MeanSyst * np.random.normal(loc=0.95, scale=0.05, size=(1,)) - MinDias) + np.min(LeftVecToSat, axis=-1, keepdims=True)
    VecToSat = np.concatenate([LeftVecToSat, RightVecToSat], axis=1)

    SatABPminSet[:, -500:] = VecToSat[:, SliceStart:SliceEnd]

    return SatABPminSet


## Reduced pulse pressure artifact
def RPP(RedABPSet):

    RedABP = RedABPSet[:, -800:]

    InitDias = np.min(RedABP, axis=-1, keepdims=True)
    eta = np.random.uniform(0.1, 0.5)
    SaledRedABP = (RedABP - InitDias) * np.linspace(eta + np.random.uniform(0.1, 0.5), eta, RedABP.shape[-1])[None] + InitDias

    RedABPSet[:, -800:] = SaledRedABP[:, :]

    return RedABPSet 


## Increased pulse pressure artifact
def IPP(IncreaseABPSet):
    IncreaseABP = IncreaseABPSet[:, -800:]

    InitDias = np.min(IncreaseABP, axis=-1, keepdims=True)
    SaledIncreaseABP = (IncreaseABP - InitDias) * np.linspace(1.0, 1.0 + np.random.uniform(0.5, 0.9), IncreaseABP.shape[-1])[None] + InitDias

    IncreaseABPSet[:, -800:] = SaledIncreaseABP[:, :]

    IncreaseABPSet = np.clip(IncreaseABPSet, 0.01, 0.99)

    return IncreaseABPSet


## High frequency artifact
def HFNoise(HFASet):
    
    LeftMinDura = 150
    RightMinDura = 70
    MaxDura = 1000

    # for left edge
    LeftNoisedIndNb = np.random.randint(0, 25, 1)
    LeftNoisedInd = np.sort(random.sample(range(2500), 2*LeftNoisedIndNb[0]))
    CandInd = np.reshape(LeftNoisedInd, (-1,2))
    SecDiff = CandInd[:,1] - CandInd[:,0] 
    LeftNoisedInd = CandInd[(SecDiff <= MaxDura) & (SecDiff >= LeftMinDura)]

    if len(LeftNoisedInd) > 0:
        LeftNoisedInd = np.concatenate(np.split(np.arange(2500), LeftNoisedInd.ravel())[1::2]).copy()
        HFASet[:,LeftNoisedInd] += np.random.normal(0, 0.05, HFASet[:,LeftNoisedInd].shape)
        HFASet[:,LeftNoisedInd] = np.clip(HFASet[:,LeftNoisedInd], 0.01, 0.99)

    # for right edge
    RightNoisedIndNb = np.random.randint(1, 11, 1)
    RightNoisedInd = np.sort([2500]+random.sample(range(2500, 3000), RightNoisedIndNb[0])*2+[3000])
    CandInd = np.reshape(RightNoisedInd, (-1,2))
    SecDiff = CandInd[:,1] - CandInd[:,0] 
    RightNoisedInd = CandInd[(SecDiff <= MaxDura) & (SecDiff >= RightMinDura)]

    if len(RightNoisedInd) < 1:
        HFASet[:,range(2500, 3000)] += np.random.normal(0, 0.1, HFASet[:,range(2500, 3000)].shape)
        HFASet[:,range(2500, 3000)] = np.clip(HFASet[:,range(2500, 3000)], 0.01, 0.99)
    else:
        RightNoisedInd = np.concatenate(np.split(np.arange(3000), RightNoisedInd.ravel())[1::2]).copy()
        HFASet[:,RightNoisedInd] += np.random.normal(0, 0.1, HFASet[:,RightNoisedInd].shape)
        HFASet[:,RightNoisedInd] = np.clip(HFASet[:,RightNoisedInd], 0.01, 0.99)

    return HFASet


## Impulse artifact
def ImpNoise(ImpSet):

    SecSize = 100
    ImpSize = 5
    fs = np.random.uniform(100, 500)

    ImpInd = np.concatenate([np.zeros(600 // SecSize - ImpSize), np.ones(ImpSize)]).astype('bool')

    VecToSat = np.arange(-(SecSize // 2), (SecSize // 2)) + 0.01
    ImpulseArt = 0.02 * np.sin((np.pi * VecToSat) / (fs * np.random.uniform(0.1, 0.4, ImpSize)[:,None])) / (np.pi * VecToSat / fs)
    ImpulseArt = np.random.choice([-1,1], size=ImpSize, replace=True)[:,None] * ImpulseArt
    ImpulseArt = np.reshape(ImpulseArt, (-1, SecSize * ImpSize)).copy()

    ImpSet[:, -(SecSize * ImpSize):] += ImpulseArt
    ImpSet = np.clip(ImpSet, 0.01, 0.99)

    return ImpSet

def LFVar(LFVSet):
    
    # generating intervals and waves
    # the number of intervals
    partition_idx = np.sort(random.sample(range(3000), np.random.randint(1,6,1)[0]))
    # end points of intervals
    ext_partition_idx = np.concatenate([[0], partition_idx, [3000]])
    intv_st_pts = ext_partition_idx[:-1]
    # calculating length
    intv_len = ext_partition_idx[1:]-ext_partition_idx[:-1]
    # generating amplitude
    intv_amp = [np.random.uniform(0.05,0.15)*(x>600) for x in intv_len]
    # calculating coefficient of x
    intv_coeff = [2*np.pi/x if x>0 else 1.0 for x in intv_len]
    
    # points on connected wave
    domain = np.array([])
    image = np.array([])
    for i in range(len(intv_len)):
        domain_tmp = np.arange(ext_partition_idx[i],ext_partition_idx[i+1])
        image_tmp = np.array([1+intv_amp[i]*np.sin(intv_coeff[i]*(x-intv_st_pts[i])) for x in domain_tmp])
        domain = np.concatenate([domain, domain_tmp])
        image = np.concatenate([image, image_tmp])

    # multiplying rates(image)
    LFVSet = image * LFVSet
    LFVSet = np.clip(LFVSet, 0.01, 0.99)

    return LFVSet

def TimeLagVar(TLVSet):

    rsp_intv_end_point = np.sort(np.append(np.random.choice(np.arange(1,2509),9,replace=False),[0,2900])) + np.arange(0, 101, 10)
    rsp_intv_len = rsp_intv_end_point[1:] - rsp_intv_end_point[:-1]

    res_diff = [5, -5, 6, -6, 7, -7, 8, -8, 9, -9]
    res_diff = np.random.permutation(res_diff)

    for j in range(10):
        sp = rsp_intv_end_point[j]
        ep = rsp_intv_end_point[j+1]
        res_tmp = signal.resample(TLVSet[:,sp:ep], rsp_intv_len[j]+res_diff[j], axis=-1)
        if j == 0:
            y_res = res_tmp
        else:
            y_res = np.concatenate([y_res, res_tmp], axis=-1)

    return y_res


# Custom loss functions
@tf.function
def AmplitudeCost(y_true, y_pred):
    SE = (y_true - y_pred)**2
    SSE = tf.reduce_sum(SE, axis=-1)
    MSSE = tf.reduce_mean(SSE)

    return MSSE #tf.reduce_mean(MaxMSEFreq)

@tf.function
def RMSE(y_true, y_pred):
    SE = (y_true - y_pred)**2
    MSE = tf.reduce_mean(SE)

    return tf.math.sqrt(MSE)

@tf.function
def DistCost(y_true, y_pred):
    AD = (tf.math.real(y_true)-tf.math.real(y_pred))**2 + (tf.math.imag(y_true)-tf.math.imag(y_pred))**2
    SAD = tf.reduce_mean(AD, axis=-1)
    MSAD = tf.reduce_mean(SAD)
    
    return MSAD