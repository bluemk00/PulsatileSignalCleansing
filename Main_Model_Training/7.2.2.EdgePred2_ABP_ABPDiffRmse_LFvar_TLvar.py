# Added one artifact type to 6.1
# Refactoring version of 7.2

#################### Setup Environment ####################
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, Permute, Reshape, GaussianNoise, LayerNormalization, LSTM, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

from lib.DataBatch import DataBatch
from lib.LossHistory import LossHistory
from lib.LossCalculator import RMSE, AmplitudeCost, DistCost

# Set CUDA environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.keras.backend.clear_session()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.98
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


#################### Helper Functions ####################
def calculate_abp_diff(ABPOut):
    return Reshape((-1,),name='ABPOut_diff')(ABPOut[:,1:] - ABPOut[:,:-1])

def calculate_amplitude(ABPOut):
    Casted = tf.cast(ABPOut, tf.complex64)
    fft = tf.signal.fft(Casted)[:, :(Casted.shape[-1]//2+1)]
    AmplitudeOut = tf.abs(fft[:,1:])*(1/(fft.shape[-1]))
    return Reshape((-1,),name='AmplitudeOut')(AmplitudeOut)

def calculate_hbt_out(sftr, sffr):
    sfhtr = (-1) * tf.cast(tf.complex(0.0,1.0), tf.complex64) * tf.cast(tf.math.sign(sffr), tf.complex64) * sftr
    return tf.math.real(tf.signal.ifft(sfhtr))


#################### Main ####################
if __name__ == "__main__":
    # Set parameters
    BatchSize = 2500
    Patience = 10000
    Epochs = 10000
    FrameSize = 50
    FrameInterval = 10
    output_mode = "ABPd"
    SaveFolder = '../4.TrainedModel/'+'7.2.2_depth'+str(FrameSize)+'_overlap'+str(FrameSize-FrameInterval)+'/'
    LatestModel = "None"


    # Load data
    TrSet = np.load('../2.ProcessedData/train_mimic_renormed.npy')[:] 
    ValSet = np.load('../2.ProcessedData/valid_vitaldb_renormed.npy')[:] 
    print('Train set total N size',TrSet.shape[0])

    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    with strategy.scope():
        # Define network architecture
        InpL = Input(shape=(3000,))
        InpFrame = tf.signal.frame(InpL, FrameSize, FrameInterval)
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
        ABPOut= Reshape((-1,),name='ABPOut')(ValDecoder)

        # Define model output
        if output_mode == "ABPd":
            ABPOut_diff = calculate_abp_diff(ABPOut)
            AEModel = Model(InpL, (ABPOut, ABPOut_diff))
            SaveFile = 'EP{epoch:d}_valABP{val_ABPOut_loss:.5f}_valABPd{val_ABPOut_diff_loss:.5f}_ABP{ABPOut_loss:.5f}_ABPd{ABPOut_diff_loss:.5f}.hdf5'
            LossSet = {'ABPOut':'mse','ABPOut_diff':RMSE}

        elif output_mode == "AMP":
            AmplitudeOut = calculate_amplitude(ABPOut)
            AEModel = Model(InpL, (ABPOut, AmplitudeOut))
            SaveFile = 'EP{epoch:d}_valABP{val_ABPOut_loss:.5f}_valAMP{val_AmplitudeOut_loss:.5f}_ABP{ABPOut_loss:.5f}_AMP{AmplitudeOut_loss:.5f}.hdf5'
            LossSet = {'ABPOut':'mse','AmplitudeOut':AmplitudeCost}

        elif output_mode == "ABPd_AMP":
            ABPOut_diff = calculate_abp_diff(ABPOut)
            AmplitudeOut = calculate_amplitude(ABPOut)
            AEModel = Model(InpL, (ABPOut, ABPOut_diff, AmplitudeOut))
            SaveFile = 'EP{epoch:d}_valABP{val_ABPOut_loss:.5f}_valABPd{val_ABPOut_diff_loss:.5f}_valAMP{val_AmplitudeOut_loss:.5f}_ABP{ABPOut_loss:.5f}_ABPd{ABPOut_diff_loss:.5f}_AMP{AmplitudeOut_loss:.5f}.hdf5'
            LossSet = {'ABPOut':'mse','ABPOut_diff':RMSE,'AmplitudeOut':AmplitudeCost}

        elif output_mode == "ABPd_HBT":
            ABPOut_diff = calculate_abp_diff(ABPOut)
            N = ABPOut.shape[-1]
            sftr = tf.signal.fft(tf.cast(ABPOut, tf.complex64))
            sffr = np.fft.fftfreq(n=N, d=1/N)
            HBTOut = tf.keras.layers.Lambda(lambda x: calculate_hbt_out(x, sffr))(sftr)
            HBTOut = Reshape((-1,),name='HBTOut')(HBTOut)
            AEModel = Model(InpL, (ABPOut, ABPOut_diff, HBTOut))
            SaveFile = 'EP{epoch:d}_valABP{val_ABPOut_loss:.5f}_valABPd{val_ABPOut_diff_loss:.5f}_valHBT{val_HBTOut_loss:.5f}_ABP{ABPOut_loss:.5f}_ABPd{ABPOut_diff_loss:.5f}_HBT{HBTOut_loss:.5f}.hdf5'
            LossSet = {'ABPOut':'mse','ABPOut_diff':RMSE,'HBTOut':DistCost}

        else:
            AEModel = Model(InpL, (ABPOut))
            SaveFile = 'EP{epoch:d}_ABP{ABPOut_loss:.5f}_valABP{val_ABPOut_loss:.5f}.hdf5'
            LossSet = {'ABPOut':'mse'}

        if LatestModel != "None":
            AEModel.load_weights(SaveFolder+LatestModel)

        # Prepare data loader
        TrainSet = DataBatch(TrSet[:], BatchSize, output_mode=output_mode)
        ValidSet = DataBatch(ValSet[:], BatchSize, output_mode=output_mode)

        # Set training configurations
        
        checkpoint = ModelCheckpoint(SaveFolder+SaveFile,monitor=('val_loss'),verbose=0, save_best_only=True, mode='auto' ,period=1) 
        earlystopper = EarlyStopping(monitor='val_loss', patience=Patience, verbose=1,restore_best_weights=True)
        history = LossHistory()
        lrate = 0.0005
        decay = 1e-6
        adam = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=decay)
        AEModel.compile(loss=LossSet, optimizer=adam) 

        # Train model
        AEModel.fit(TrainSet, validation_data = (ValidSet), verbose=1, epochs=Epochs, callbacks=[history,earlystopper,checkpoint])
        