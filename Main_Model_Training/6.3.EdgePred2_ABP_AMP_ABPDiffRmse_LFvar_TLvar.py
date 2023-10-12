from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal

import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense,Concatenate, Permute, Activation, MaxPooling2D, Conv2D, Flatten,Dot,Reshape,MaxPooling1D,Conv1D, GaussianNoise, LayerNormalization,UpSampling1D, Conv1DTranspose, GlobalAveragePooling2D, LSTM, Bidirectional, GaussianNoise, TimeDistributed
import tensorflow as tf
from tensorflow.keras.backend import  expand_dims

from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model ,load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K,losses
from tensorflow.keras.activations import softmax



class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self,logs={}):
        self.losses=[]
        self.val_losses=[]
        
    def on_epoch_end(self,batch,logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        

class DataBatch(tf.keras.utils.Sequence):
    
    def __init__(self, DataInp, BatchSize, shuffle = False, *args, **kwargs):
        
        self.DataInp = DataInp
        self.DataLen = DataInp.shape[0]
        self.TimeLen = DataInp.shape[1]

        self.BatchSize = BatchSize
                
        self.Indices = np.arange(self.DataLen)
        self.shuffle = shuffle
        
                
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
        NoiseRate = [WNR*0.2, WNR*0.2, WNR*0.2, WNR*0.2, WNR*0.2]
        NoiseSize = [int(len(BatchIndex)*NR) for NR in NoiseRate]
        NormalSize = len(BatchIndex) - sum(NoiseSize)
        NoiseBatchInd = np.concatenate([np.ones(i)*num for num,i in enumerate([NormalSize]+NoiseSize)]) # 0 : normal
        NoiseBatchInd = np.array([int(ind+10*(num%3)) for num, ind in enumerate(NoiseBatchInd)])
        NoiseBatchInd = np.random.permutation(NoiseBatchInd)

        # select for saturation to ABP maximum
        SelectedBatch[NoiseBatchInd%10==1] = self.SatABPmax(SelectedBatch[NoiseBatchInd%10==1])

        # select for saturation to ABP minmum
        SelectedBatch[NoiseBatchInd%10==2] = self.SatABPmin(SelectedBatch[NoiseBatchInd%10==2])

        # Reduced pulse pressure artifact
        SelectedBatch[NoiseBatchInd%10==3] = self.RPP(SelectedBatch[NoiseBatchInd%10==3])

        # High frequency artifact
        SelectedBatch[NoiseBatchInd%10==4] = self.HFNoise(SelectedBatch[NoiseBatchInd%10==4])

        # Impulse artifact
        SelectedBatch[NoiseBatchInd%10==5] = self.ImpNoise(SelectedBatch[NoiseBatchInd%10==5])
        
        # Global Low frequency variance
        SelectedBatch[NoiseBatchInd<10] = self.LFVar(SelectedBatch[NoiseBatchInd<10])
        
        # Global Time lag variance
        SelectedBatch[NoiseBatchInd>=20] = self.TimeLagVar(SelectedBatch[NoiseBatchInd>=20])
        
        InpData = SelectedBatch.copy()
              
        ### FFT Oup
        OutCasted = tf.cast(OutpData, tf.complex64) 
        Outfft = tf.signal.fft(OutCasted)[:, :(OutCasted.shape[-1]//2+1)]
        OutAmplitude = tf.abs(Outfft[:,1:])*(1/(Outfft.shape[-1])) # 상대적 진폭 Relative amplitude
#         print('1',OutAmplitude.shape)
        
        return (InpData), (OutpData, OutpData_diff, OutAmplitude) # (InpData), (OutpData) 

    
        
    def on_epoch_end(self,):
        self.Indices = np.arange(self.DataLen)
        if self.shuffle:
            np.random.shuffle(self.Indices)
            
            
            
    ## saturation to ABP maximum artifact
    def SatABPmax(self,SatABPmaxSet):

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
    def SatABPmin(self,SatABPminSet):

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
    def RPP(self,RedABPSet):

        RedABP = RedABPSet[:, -500:]

        InitDias = np.min(RedABP, axis=-1, keepdims=True)
        eta = np.random.uniform(0.1, 0.5)
        SaledRedABP = (RedABP - InitDias) * np.linspace(eta + np.random.uniform(0.1, 0.5), eta, RedABP.shape[-1])[None] + InitDias

        RedABPSet[:, -500:] = SaledRedABP[:, :]

        return RedABPSet 



    ## High frequency artifact
    def HFNoise(self,HFASet):
        
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
    def ImpNoise(self,ImpSet):

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

    def LFVar(self,LFVSet):
        
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
    
    def TimeLagVar(self, TLVSet):
    
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
    


    
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"



# TensorFlow wizardry
config = tf.compat.v1.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.98
# Create a session with the above options specified.
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))     



if __name__ == "__main__":

    BatchSize = 500

    ## Data selection
    TrSet = np.load('../2.ProcessedData/train_mimic_renormed.npy')[:] 
    ValSet = np.load('../2.ProcessedData/valid_vitaldb_renormed.npy')[:] 
    
    print('Train set total N size',TrSet.shape[0])
    

    strategy = tf.distribute.MirroredStrategy( cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()) 
    
    with strategy.scope():

        FrameSize = 50
        OverlapSize = 10 # actually, frame interval

        InpL = Input(shape=(3000,))

        InpFrame = tf.signal.frame(InpL, FrameSize, OverlapSize)
        InpFrameNoise = tf.keras.layers.GaussianNoise(0.0)(InpFrame)

        Encoder = Dense(50, activation='relu', name = 'Encoder1')(InpFrameNoise)
        Encoder = Dropout(0.0)(Encoder, training=True)

        Encoder = LSTM(50, return_sequences=True)(Encoder)
        Att_front = LSTM(10, return_sequences=True)(Encoder)
        Att_front = LayerNormalization(axis=(1), epsilon=0.001)(Att_front)
        Att_front = Permute((2,1))(Att_front)
        Att_front = Dropout(0.0)(Att_front, training=True)
        Att_front = Dense(InpFrame.shape[1],activation='softmax')(Att_front)
        Att_front = Permute((2,1), name='Att_front')(Att_front)

        Context = InpFrameNoise[:,:,None] * Att_front[:,:,:,None]
        Context = tf.reduce_sum(Context, axis=(1), name = 'Context')

        Decoder = LSTM(50, return_sequences=True)(Context)
        Decoder = LayerNormalization(axis=(1,2), epsilon=0.001)(Decoder)

        Att_back = LSTM(500, return_sequences=False)(InpFrame)
        Att_back = Reshape((10, 50))(Att_back)
        Att_back = LayerNormalization(axis=(1,2), epsilon=0.001)(Att_back)
        Att_back = LSTM(50, return_sequences=True)(Att_back)
        Att_back = Dropout(0.0)(Att_back, training=True)
        Att_back = Dense(50,activation='tanh')(Att_back)


        Scaling = Decoder + Att_back
        Scaling = LSTM(50, return_sequences=True, name = 'Scaling')(Scaling)

        ValDecoder = Context + Scaling
        ABPOut= Reshape((-1,),name='ABPOut')(ValDecoder)
        ABPOut_diff = Reshape((-1,),name='ABPOut_diff')(ABPOut[:,1:]-ABPOut[:,:-1])

        Casted = tf.cast(ABPOut, tf.complex64) 
        fft = tf.signal.fft(Casted)[:, :(Casted.shape[-1]//2+1)]
        AmplitudeOut = tf.abs(fft[:,1:])*(1/(fft.shape[-1])) 
        AmplitudeOut= Reshape((-1,),name='AmplitudeOut')(AmplitudeOut)

        AEModel = Model(InpL, (ABPOut, ABPOut_diff, AmplitudeOut))
#         AEModel.load_weights('./6.3.EdgePred2_ABP_AMP_ABPDiffRmse_LFvar_TLvar/EdgePred2_197_0.00242_0.00344_0.00021_0.00032.hdf5')
        
        def AmplitudeCost(y_true, y_pred):
            SE = (y_true - y_pred)**2
            SSE = tf.reduce_sum(SE, axis=-1)
            MSSE = tf.reduce_mean(SSE)

            return MSSE #tf.reduce_mean(MaxMSEFreq)
        
        
        def RMSE(y_true, y_pred):
            SE = (y_true - y_pred)**2
            MSE = tf.reduce_mean(SE)

            return tf.math.sqrt(MSE)
        
        

        # Data Loader
        TrainSet = DataBatch(TrSet[:], BatchSize)
        ValidSet = DataBatch(ValSet[:], BatchSize)

        SaveFilePath = '../4.TrainedModel/6.3_depth50_overlap40/6.3.EdgePred2_{epoch:d}_AMP{AmplitudeOut_loss:.5f}_valAMP{val_AmplitudeOut_loss:.5f}_ABPd{ABPOut_diff_loss:.5f}_valABPd{val_ABPOut_diff_loss:.5f}_ABP{ABPOut_loss:.5f}_valABP{val_ABPOut_loss:.5f}.hdf5'
        # SaveFilePath = './41.EdgePred2_ABP_AMP_ABPdiff_mse/EdgePred2_{epoch:d}_{AmplitudeOut_loss:.5f}_{val_AmplitudeOut_loss:.5f}_{ABPOut_diff_loss:.5f}_{val_ABPOut_diff_loss:.5f}_{ABPOut_loss:.5f}_{val_ABPOut_loss:.5f}.hdf5'
        checkpoint = ModelCheckpoint(SaveFilePath,monitor=('val_loss'),verbose=0, save_best_only=True, mode='auto' ,period=1) 
        earlystopper = EarlyStopping(monitor='val_loss', patience=1000, verbose=1,restore_best_weights=True)
        history = LossHistory()

        
        lrate = 0.0005
        decay = 1e-6
        adam = tf.keras.optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=decay)
        AEModel.compile(loss={'AmplitudeOut':AmplitudeCost,'ABPOut_diff':RMSE,'ABPOut':'mse'}, optimizer=adam) #loss={'AmplitudeOut':AmplitudeCost,'ABPOut':'mse'}

        AEModel.fit(TrainSet, validation_data = (ValidSet), verbose=1, epochs=10000, callbacks=[history,earlystopper,checkpoint]) #class_weight=class_weight
        