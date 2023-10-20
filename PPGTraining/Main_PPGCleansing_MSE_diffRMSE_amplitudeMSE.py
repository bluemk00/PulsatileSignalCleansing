import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, Permute, Reshape, LayerNormalization, LSTM, Bidirectional, GaussianNoise

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

    
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

sys.path.append("./lib/")
from utils import *

tf.keras.backend.clear_session()

# TensorFlow wizardry
config = tf.compat.v1.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.98
# Create a session with the above options specified.
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))     


outptype = 2
'''
outptype
0 : (OutpData, OutpData_diff)
1 : (OutpData, OutAmplitude)
2 : (OutpData, OutpData_diff, OutAmplitude)
3 : (OutpData, shtr)
4 : (OutpData, OutpData_diff, shtr)
'''

if __name__ == "__main__":

    BatchSize = 1500

    ## Data selection
    TrSet = np.load('../2.ProcessedData/MIMIC_PPG_clean_30s_total/MIMIC_PPG_TrSet.npy')[:]
    ValSet = np.load('../2.ProcessedData/MIMIC_PPG_clean_30s_total/MIMIC_PPG_ValSet.npy')[:]

    print('************************************************')
    print(f'    Train set total {TrSet.shape[0]} size   ')
    print(f'    Valid set total {ValSet.shape[0]} size   ')
    print('************************************************')
    

    strategy = tf.distribute.MirroredStrategy( cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()) 
    
    with strategy.scope():

        FrameSize = 50
        OverlapSize = 10 # actually, frame interval

        InpL = Input(shape=(3000,))

        InpFrame = tf.signal.frame(InpL, FrameSize, OverlapSize)
        InpFrameNoise = GaussianNoise(0.0)(InpFrame)

        Encoder = Dense(50, activation='relu', name = 'Encoder1')(InpFrameNoise)
        Encoder = Dropout(0.0)(Encoder, training=True)

        Encoder = Bidirectional(LSTM(25, return_sequences=True))(Encoder)
        Att_front = Bidirectional(LSTM(5, return_sequences=True))(Encoder)
        Att_front = LayerNormalization(axis=(1), epsilon=0.001)(Att_front)
        Att_front = Permute((2,1))(Att_front)
        Att_front = Dropout(0.0)(Att_front, training=True)
        Att_front = Dense(InpFrame.shape[1],activation='softmax')(Att_front)
        Att_front = Permute((2,1), name='Att_front')(Att_front)

        Context = InpFrameNoise[:,:,None] * Att_front[:,:,:,None]
        Context = tf.reduce_sum(Context, axis=(1), name = 'Context')

        Decoder = Bidirectional(LSTM(25, return_sequences=True))(Context)
        Decoder = LayerNormalization(axis=(1,2), epsilon=0.001)(Decoder)

        Att_back = Bidirectional(LSTM(250, return_sequences=False))(InpFrame)
        Att_back = Reshape((10, 50))(Att_back)
        Att_back = LayerNormalization(axis=(1,2), epsilon=0.001)(Att_back)
        Att_back = Bidirectional(LSTM(25, return_sequences=True))(Att_back)
        Att_back = Dropout(0.0)(Att_back, training=True)
        Att_back = Dense(50,activation='tanh')(Att_back)

        Scaling = Decoder + Att_back
        Scaling = Bidirectional(LSTM(25, return_sequences=True, name = 'Scaling'))(Scaling)

        ValDecoder = Context + Scaling
        PPGOut= Reshape((-1,),name='PPGOut')(ValDecoder)
        PPGOut_diff = Reshape((-1,),name='PPGOut_diff')(PPGOut[:,1:]-PPGOut[:,:-1])

        if (outptype == 1) or (outptype == 2):
            Casted = tf.cast(PPGOut, tf.complex64) 
            fft = tf.signal.fft(Casted)[:, :(Casted.shape[-1]//2+1)]
            AmplitudeOut = tf.abs(fft[:,1:])*(1/(fft.shape[-1])) 
            AmplitudeOut= Reshape((-1,),name='AmplitudeOut')(AmplitudeOut)

        def tmp (sftr):
            sfhtr = (-1) * tf.cast(tf.complex(0.0,1.0), tf.complex64) * tf.cast(tf.math.sign(sffr), tf.complex64) * sftr
            return tf.math.real(tf.signal.ifft(sfhtr))
        
        if (outptype == 3) or (outptype == 4):
            N = PPGOut.shape[-1]
            sftr = tf.signal.fft(tf.cast(PPGOut, tf.complex64))
            sffr = np.fft.fftfreq(n=N, d=1/N)
            HBTOut = tf.keras.layers.Lambda(tmp)(sftr)
            HBTOut = Reshape((-1,),name='HBTOut')(HBTOut)

        if outptype == 0:
            AEModel = Model(InpL, (PPGOut, PPGOut_diff))
            loss_set = {'PPGOut':'mse', 'PPGOut_diff':RMSE}
            SaveFolder = './models/Main_PPGCleansing_MSE_diffRMSE/'
            SaveFilePath = 'AE_PPGPPGdAMP_{epoch:04}_val{val_loss:.7f}_valPPG{val_PPGOut_loss:.7f}_valPPGd{val_PPGOut_diff_loss:.7f}.hdf5'
        elif outptype == 1:
            AEModel = Model(InpL, (PPGOut, AmplitudeOut))
            loss_set = {'PPGOut':'mse', 'AmplitudeOut':AmplitudeCost}
            SaveFolder = './models/Main_PPGCleansing_MSE_amplitudeMSE/'
            SaveFilePath = 'AE_PPGPPGdAMP_{epoch:04}_val{val_loss:.7f}_valPPG{val_PPGOut_loss:.7f}}_valAMP{val_AmplitudeOut_loss:.7f}.hdf5'
        elif outptype == 2:
            AEModel = Model(InpL, (PPGOut, PPGOut_diff, AmplitudeOut))
            loss_set = {'PPGOut':'mse', 'PPGOut_diff':RMSE, 'AmplitudeOut':AmplitudeCost}
            SaveFolder = './models/Main_PPGCleansing_MSE_diffRMSE_amplitudeMSE/'
            SaveFilePath = 'AE_PPGPPGdAMP_{epoch:04}_val{val_loss:.7f}_valPPG{val_PPGOut_loss:.7f}_valPPGd{val_PPGOut_diff_loss:.7f}_valAMP{val_AmplitudeOut_loss:.7f}.hdf5'
        elif outptype == 3:
            AEModel = Model(InpL, (PPGOut, HBTOut))
            loss_set = {'PPGOut':'mse', 'HBTOut':DistCost}
            SaveFolder = './models/Main_PPGCleansing_MSE_HBTMSE/'
            SaveFilePath = 'AE_PPGHBT_{epoch:04}_val{val_loss:.7f}_valPPG{val_PPGOut_loss:.7f}_valHBT{val_HBTOut_loss:.7f}.hdf5'
        elif outptype == 4:
            AEModel = Model(InpL, (PPGOut, PPGOut_diff, HBTOut))
            loss_set = {'PPGOut':'mse', 'PPGOut_diff':RMSE, 'HBTOut':DistCost}
            SaveFolder = './models/Main_PPGCleansing_MSE_diffRMSE_HBTMSE/'
            SaveFilePath = 'AE_PPGPPGdHBT_{epoch:04}_val{val_loss:.7f}_valPPG{val_PPGOut_loss:.7f}_valPPGd{val_PPGOut_diff_loss:.7f}_valHBT{val_HBTOut_loss:.7f}.hdf5'

        # Model Loader
        # ModelPath = ''
        # AEModel.load_weights(ModelPath+'7.5.EdgePred2_5851_PPG0.00073_valPPG0.00089_PPGd0.00630_valPPGd0.00681_HBT0.00066_valHBT0.00083.hdf5')
        

        # Data Loader
        TrainSet = DataBatch(TrSet[:], BatchSize, outptype=outptype)
        ValidSet = DataBatch(ValSet[:], BatchSize, outptype=outptype)

        checkpoint = ModelCheckpoint(SaveFolder + SaveFilePath, monitor=('val_loss'), verbose=0, save_best_only=True, mode='auto', period=1) 
        earlystopper = EarlyStopping(monitor='val_loss', patience=10000, verbose=1, restore_best_weights=True)
        history = LossHistory()

        
        lrate = 0.0005
        decay = 1e-6
        adam = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=decay)
        AEModel.compile(loss=loss_set, optimizer=adam)

        AEModel.fit(TrainSet, validation_data = (ValidSet), verbose=1, epochs=10000, callbacks=[history,earlystopper,checkpoint]) #class_weight=class_weight
        