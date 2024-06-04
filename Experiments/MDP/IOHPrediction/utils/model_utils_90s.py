import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers, Input, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Activation, Dropout
import tensorflow_probability as tfp
from tensorflow.keras import backend as K

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# TensorFlow wizardry
config = tf.compat.v1.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.98
# Create a session with the above options specified.
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

class DoGenVec(tf.keras.layers.Layer):
    
    def __init__(self, OutDimes):
        super(DoGenVec, self).__init__()
        self.Shapelet1Size = OutDimes[0]
        self.WinSize = OutDimes[1]
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'Shapelet1Size': self.Shapelet1Size,
            'WinSize': self.WinSize})
        return config
    
    def build(self, input_shape):
        self.GenVec = self.add_weight("GenVec", shape=[self.Shapelet1Size, self.WinSize], initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01))
    
    def call(self, input):
        input = K.sum(input) * 0 + 1 
        return (input * self.GenVec)
    
b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
N = int(np.ceil((4 / b)))
if not N % 2:  # Make sure that N is odd.
    N += 1  
RanVar = tf.constant(np.arange(N), dtype=tf.float32)

def FilterGen(FC):
    # Sinc function
    X = (2 * FC * (RanVar[None] - (N - 1) / 2))
    X = tf.where(X == 0, K.epsilon(), X)
    SinF = tf.sin(np.pi * X) / (np.pi * X)

    # Blackman window
    BW = 0.42 - 0.5 * tf.math.cos(2 * np.pi * RanVar / (N - 1)) + 0.08 * tf.math.cos(4 * np.pi * RanVar / (N - 1))

    SinFBW = SinF * BW
    LP = SinFBW / K.sum(SinFBW, axis=-1, keepdims=True)
    HP = -LP
    TmpZeros = tf.zeros((N - 1) // 2)
    TmpOnes = tf.ones(1)
    AddOne = tf.concat([TmpZeros, TmpOnes, TmpZeros], axis=0)
    HP += AddOne

    return LP, HP

def DownSampling(ToDown):
    if ToDown.shape[1] % 2 != 0: 
        ToDown = tf.concat([K.mean(ToDown[:, :2], axis=1, keepdims=True), ToDown[:, 2:]], axis=1)
    return K.mean(tf.signal.frame(ToDown[:, :, 0], 2, 2, axis=1), axis=-1)

def create_model(input_shape):
    Shapelet1Size = 30
    PaddSideForSim = 12
    FrameSize = 50 
    AttSize = 5
    b = 0.08 # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
    N = int(np.ceil((4 / b)))
    if not N % 2:  # Make sure that N is odd.
        N += 1  
    
    InputVec = Input(shape=(9000), name='Input')
    
    ### A. Discrete wavelet transform layers ###
    
    # DWT 1st level
    FC1 = Conv1D(filters= 1,kernel_size= 20, strides=20//5, activation='softplus')(InputVec[:,:,None])
    FC1 = MaxPooling1D(pool_size=20,strides=20//5)(FC1)
    FC1 = Conv1D(filters= 1,kernel_size= 10, strides=10//2)(FC1)
    FC1 = MaxPooling1D(pool_size=10,strides=10//2)(FC1)[:,:,0]
    FC1 = Dense(10, activation='relu')(FC1)
    FC1 = Dense(1, activation='sigmoid')(FC1)
    FC1 = FC1*(0.5-K.epsilon())+K.epsilon()
    LP1, HP1 =  FilterGen (FC1)
    
    InputVecPad = tf.signal.frame(InputVec, N, 1)
    LP1_res = K.sum(InputVecPad * LP1[:,None,:], axis=-1, keepdims=True)
    LP1_Down = DownSampling(LP1_res)
    
    
    # DWT 2nd level
    FC2 = Conv1D(filters= 1,kernel_size= 20, strides=20//5, activation='softplus')(LP1_Down[:,:,None])
    FC2 = MaxPooling1D(pool_size=20,strides=20//5)(FC2)
    FC2 = Conv1D(filters= 1,kernel_size= 10, strides=10//2)(FC2)
    FC2 = MaxPooling1D(pool_size=10,strides=10//2)(FC2)[:,:,0]
    FC2 = Dense(10, activation='relu')(FC2)
    FC2 = Dense(1, activation='sigmoid')(FC2)
    FC2 = FC2*(0.5-K.epsilon())+K.epsilon()
    LP2, HP2 =  FilterGen (FC2)
    
    LP1_DownPad = tf.signal.frame(LP1_Down, N, 1)
    LP2_res = K.sum(LP1_DownPad * LP2[:,None,:], axis=-1, keepdims=True)
    LP2_Down = DownSampling(LP2_res)
    
    
    # DWT 3rd level
    FC3 = Conv1D(filters= 1,kernel_size= 10, strides=10//4, activation='softplus')(LP2_Down[:,:,None])
    FC3 = MaxPooling1D(pool_size=10,strides=10//4)(FC3)
    FC3 = Conv1D(filters= 1,kernel_size= 10, strides=10//2)(FC3)
    FC3 = MaxPooling1D(pool_size=10,strides=10//2)(FC3)[:,:,0]
    FC3 = Dense(10, activation='relu')(FC3)
    FC3 = Dense(1, activation='sigmoid')(FC3)
    FC3 = FC3*(0.5-K.epsilon())+K.epsilon()
    LP3, HP3 =  FilterGen (FC3)
    
    LP2_DownPad = tf.signal.frame(LP2_Down, N, 1)
    LP3_res = K.sum(LP2_DownPad * LP3[:,None,:], axis=-1, keepdims=True)
    LP3_Down = DownSampling(LP3_res)
    
    
    # DWT 4th level
    FC4 = Conv1D(filters= 1,kernel_size= 10, strides=10//2, activation='softplus')(LP3_Down[:,:,None])
    FC4 = MaxPooling1D(pool_size=10,strides=10//2)(FC4)
    FC4 = Conv1D(filters= 1,kernel_size= 5, strides=5//2)(FC4)
    FC4 = MaxPooling1D(pool_size=5,strides=5//2)(FC4)[:,:,0]
    FC4 = Dense(10, activation='relu')(FC4)
    FC4 = Dense(1, activation='sigmoid')(FC4)
    FC4 = FC4*(0.5-K.epsilon())+K.epsilon()
    LP4, HP4 =  FilterGen (FC4)
    
    LP3_DownPad = tf.signal.frame(LP3_Down, N, 1)
    LP4_res = K.sum(LP3_DownPad * LP4[:,None,:], axis=-1, keepdims=True)
    LP4_Down = DownSampling(LP4_res)
    
    
    # DWT 5th level
    FC5 = Conv1D(filters= 1,kernel_size= 10, strides=10//2, activation='softplus')(LP4_Down[:,:,None])
    FC5 = MaxPooling1D(pool_size=10,strides=10//2)(FC5)
    FC5 = Conv1D(filters= 1,kernel_size= 5, strides=5//2)(FC5)
    FC5 = MaxPooling1D(pool_size=5,strides=5//2)(FC5)[:,:,0]
    FC5 = Dense(10, activation='relu')(FC5)
    FC5 = Dense(1, activation='sigmoid')(FC5)
    FC5 = FC5*(0.5-K.epsilon())+K.epsilon()
    LP5, HP5 =  FilterGen (FC5)
    
    LP4_DownPad = tf.signal.frame(LP4_Down, N, 1)
    LP5_res = K.sum(LP4_DownPad * LP5[:,None,:], axis=-1, keepdims=True)
    LP5_Down = DownSampling(LP5_res)
    
    LP5_FeatDim = tf.signal.frame(LP5_Down, FrameSize, 1)
    
    
    ### B. Local shape similarity layers ###
    GenVecLayer = DoGenVec([Shapelet1Size, FrameSize])
    GenVec = Activation('sigmoid')(GenVecLayer(InputVec)) 
    
    LP5_X_sqare = K.sum(K.square(LP5_FeatDim), axis=2,keepdims=True)
    LP5_Y_sqare = K.sum(K.square(GenVec[:]), axis=1)[None,None]
    LP5_XY = tf.matmul(LP5_FeatDim, GenVec[:], transpose_b=True)
    LP5_Dist = (LP5_X_sqare + LP5_Y_sqare - 2*LP5_XY) 
    LP5_Dist_sqrt = K.sqrt(LP5_Dist+K.epsilon())
    
    
    ### C. Local imporatnce layers ###
    
    # Interval-Wise importance and select interval
    LP5_ATT = Conv1D(filters= 1,kernel_size= FrameSize, activation='softplus', strides=1,padding="same")(Dropout(0.0)(LP5_Down[:,:,None]))
    LP5_ATT = MaxPooling1D(pool_size=FrameSize//2,strides=FrameSize//4)(LP5_ATT)
    LP5_ATT = Dense(30, activation='relu')(LP5_ATT[:,:,0])
    LP5_Mus = Dense(AttSize, activation='sigmoid')(LP5_ATT)
    
    dist = tfp.distributions.Normal( LP5_Mus[:,:,None], 0.075, name='Normal') 
    RandVec = tf.constant(np.linspace(0, 1, LP5_FeatDim.shape[1]), dtype=tf.float32)[None,None]
    RandVec = tf.tile(RandVec, (K.shape(LP5_Mus)[0], AttSize, 1))
    KerVal = dist.prob(RandVec)
    MinKV = K.min(KerVal, axis=-1, keepdims=True)
    MaxKV = K.max(KerVal, axis=-1, keepdims=True)
    KvRate = (KerVal - MinKV)/(MaxKV - MinKV)
    AggKvRate = K.sum(KvRate, axis=1)
    
    
    ### D. Regresion layers ###
    Features_W = K.exp(-LP5_Dist_sqrt) * AggKvRate[:,:,None]
    Features = K.max(Features_W, axis=1) 
    BinOut = Dense(1, activation='sigmoid',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=0),name='BinOut')(Dropout(0.0)(Features))
    
    BinModel = Model(InputVec, BinOut)
    
    
    ### Add cluster loss ###
    LP5_Dist_exp = tf.tile(LP5_Dist[:,None], (1,AttSize,1,1)) # Batch, AttNum, Interval, Centroid
    KvRate_MaxInd = K.argmax(KvRate, axis=-1) 
    Att_Loss = K.min(tf.gather_nd(LP5_Dist_exp,KvRate_MaxInd[:,:,None], batch_dims=2 ), axis=-1)
    ClLoss = K.mean(Att_Loss * K.max(KvRate, axis=-1))
    
    BinModel.add_loss(ClLoss)
    BinModel.add_metric(ClLoss, name='ClLoss')


    return BinModel