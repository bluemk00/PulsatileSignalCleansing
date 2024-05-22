import subprocess
import random
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import backend as K

# Ensure scipy is installed
try:
    from scipy import signal
except ImportError:
    print("scipy is not installed. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    from scipy import signal

class LossHistory(tf.keras.callbacks.Callback):
    """Callback to log loss history to a file."""

    def __init__(self, filename):
        self.filename = filename

    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []
        with open(self.filename, 'w') as f:
            f.write("Epoch,Loss,Validation Loss\n")

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        with open(self.filename, 'a') as f:
            f.write(f"{epoch},{logs.get('loss')},{logs.get('val_loss')}\n")


class DataBatch(tf.keras.utils.Sequence):
    """Custom data generator for Keras."""

    def __init__(self, DataInp, BatchSize, shuffle=False, outptype=0, *args, **kwargs):
        self.DataInp = DataInp
        self.DataLen = DataInp.shape[0]
        self.TimeLen = DataInp.shape[1]
        self.BatchSize = BatchSize
        self.Indices = np.arange(self.DataLen)
        self.shuffle = shuffle
        self.outptype = outptype
        """
        outptype:
        0 : (OutpData, OutpData_diff)
        1 : (OutpData, OutAmplitude)
        2 : (OutpData, OutpData_diff, OutAmplitude)
        """

    def __len__(self):
        return math.ceil(self.DataLen / self.BatchSize)

    def __getitem__(self, idx):
        BatchIndex = self.Indices[idx*self.BatchSize:(idx+1)*self.BatchSize]
        SelectedBatch = self.DataInp[BatchIndex]
        
        # Copying high-quality data as output
        OutpData = SelectedBatch[:, -500:].copy()
        OutpData_diff = OutpData[:, 1:] - OutpData[:, :-1]
        
        # Generate noise sampling index
        WNR = 0.85  # Whole noise rate
        NoiseRate = [WNR*0.23, WNR*0.23, WNR*0.18, WNR*0.18, WNR*0.18]
        NoiseSize = [int(len(BatchIndex)*NR) for NR in NoiseRate]
        NormalSize = len(BatchIndex) - sum(NoiseSize)
        NoiseBatchInd = np.concatenate([np.ones(i)*num for num, i in enumerate([NormalSize]+NoiseSize)])
        NoiseBatchInd = np.array([int(ind+10) if num%4==1 else int(ind+20) if num%4==2 else int(ind+30) if num%4==3 else int(ind) for num, ind in enumerate(NoiseBatchInd)])
        NoiseBatchInd = np.random.permutation(NoiseBatchInd)

        # Apply different types of noise artifacts
        SelectedBatch[NoiseBatchInd % 10 == 1] = self.SatMax(SelectedBatch[NoiseBatchInd % 10 == 1])
        SelectedBatch[NoiseBatchInd % 10 == 2] = self.SatMin(SelectedBatch[NoiseBatchInd % 10 == 2])
        SelectedBatch[NoiseBatchInd % 10 == 3] = self.RedPP(SelectedBatch[NoiseBatchInd % 10 == 3])
        SelectedBatch[NoiseBatchInd % 10 == 4] = self.AmpPP(SelectedBatch[NoiseBatchInd % 10 == 4])
        SelectedBatch[NoiseBatchInd % 10 == 5] = self.Impulse(SelectedBatch[NoiseBatchInd % 10 == 5])
        SelectedBatch[NoiseBatchInd < 10] = self.LowFreq(SelectedBatch[NoiseBatchInd < 10])
        mask = (NoiseBatchInd >= 10) & (NoiseBatchInd < 20)
        SelectedBatch[mask] = self.HighFreq(SelectedBatch[mask])
        SelectedBatch[NoiseBatchInd > 29] = self.TimeLag(SelectedBatch[NoiseBatchInd > 29])
        
        InpData = SelectedBatch.copy()

        # FFT output
        if (self.outptype == 1) or (self.outptype == 2):
            OutCasted = tf.cast(OutpData, tf.complex64)
            Outfft = tf.signal.fft(OutCasted)[:, :(OutCasted.shape[-1] // 2 + 1)]
            OutAmplitude = tf.abs(Outfft[:, 1:]) * (1 / (Outfft.shape[-1]))

        if self.outptype == 0:
            return InpData, (OutpData, OutpData_diff)
        elif self.outptype == 1:
            return InpData, (OutpData, OutAmplitude)
        elif self.outptype == 2:
            return InpData, (OutpData, OutpData_diff, OutAmplitude)

    def on_epoch_end(self):
        self.Indices = np.arange(self.DataLen)
        if self.shuffle:
            np.random.shuffle(self.Indices)

    # Saturation to maximum artifact
    def SatMax(self, SatMaxSet):
        SatMaxEnd = np.random.randint(501, 1000)
        SliceStart = np.random.randint(0, SatMaxEnd - 500)
        SliceEnd = SliceStart + 500
        MinDias = np.min(SatMaxSet[:, -1000:-500], axis=-1, keepdims=True)
        VecToSat = np.arange(SatMaxEnd)
        TmpMax = np.random.uniform(low=0.7, high=0.95, size=(1,))
        VecToSat = np.tanh((np.pi * VecToSat * np.random.uniform(low=0.1, high=0.9, size=(1,))) / 100) * (TmpMax - MinDias) + MinDias
        SatMaxSet[:, -500:] = VecToSat[:, SliceStart:SliceEnd]
        return SatMaxSet

    # Saturation to minimum artifact
    def SatMin(self, SatMinSet):
        SatMinEnd = np.random.randint(501, 1000)
        SliceStart = np.random.randint(0, SatMinEnd - 500)
        SliceEnd = SliceStart + 500
        MinDias = np.min(SatMinSet[:, -1000:-500], axis=-1, keepdims=True) * np.random.uniform(low=0.7, high=0.99, size=(1,))
        MeanSyst = np.mean(SatMinSet[:, -600:-500], axis=-1, keepdims=True)
        LeftSize = int(SatMinEnd * 0.8)
        RightSize = SatMinEnd - LeftSize
        LeftVecToSat = np.arange(LeftSize)
        RightVecToSat = np.arange(RightSize)
        LeftVecToSat = (1 - np.tanh((np.pi * LeftVecToSat * np.random.uniform(low=0.1, high=0.3, size=(1,))) / 100)) * (MeanSyst * np.random.normal(loc=1.05, scale=0.05, size=(1,)) - MinDias) + MinDias
        RightVecToSat = (np.tanh((np.pi * (RightVecToSat - RightSize) * np.random.uniform(low=0.7, high=0.99, size=(1,))) / 100) + 1) * (MeanSyst * np.random.normal(loc=0.95, scale=0.05, size=(1,)) - MinDias) + np.min(LeftVecToSat, axis=-1, keepdims=True)
        VecToSat = np.concatenate([LeftVecToSat, RightVecToSat], axis=1)
        SatMinSet[:, -500:] = VecToSat[:, SliceStart:SliceEnd]
        return SatMinSet

    # Reduced pulse pressure artifact
    def RedPP(self, RedPPSet):
        RedABP = RedPPSet[:, -800:]
        InitDias = np.min(RedABP, axis=-1, keepdims=True)
        eta = np.random.uniform(0.1, 0.5)
        SaledRedABP = (RedABP - InitDias) * np.linspace(eta + np.random.uniform(0.1, 0.5), eta, RedABP.shape[-1])[None] + InitDias
        RedPPSet[:, -800:] = SaledRedABP
        return RedPPSet

    # Amplifying pulse pressure artifact
    def AmpPP(self, AmpPPSet):
        IncreaseABP = AmpPPSet[:, -800:]
        InitDias = np.min(IncreaseABP, axis=-1, keepdims=True)
        SaledIncreaseABP = (IncreaseABP - InitDias) * np.linspace(1.0, 1.0 + np.random.uniform(0.5, 0.9), IncreaseABP.shape[-1])[None] + InitDias
        AmpPPSet[:, -800:] = SaledIncreaseABP
        AmpPPSet = np.clip(AmpPPSet, 0.01, 0.99)
        return AmpPPSet

    # High frequency noise artifact
    def HighFreq(self, HFSet):
        LeftMinDura = 150
        RightMinDura = 70
        MaxDura = 1000
        LeftNoisedIndNb = np.random.randint(0, 25, 1)
        LeftNoisedInd = np.sort(random.sample(range(2500), 2*LeftNoisedIndNb[0]))
        CandInd = np.reshape(LeftNoisedInd, (-1, 2))
        SecDiff = CandInd[:, 1] - CandInd[:, 0]
        LeftNoisedInd = CandInd[(SecDiff <= MaxDura) & (SecDiff >= LeftMinDura)]
        if len(LeftNoisedInd) > 0:
            LeftNoisedInd = np.concatenate(np.split(np.arange(2500), LeftNoisedInd.ravel())[1::2]).copy()
            HFSet[:, LeftNoisedInd] += np.random.normal(0, 0.05, HFSet[:, LeftNoisedInd].shape)
            HFSet[:, LeftNoisedInd] = np.clip(HFSet[:, LeftNoisedInd], 0.01, 0.99)
        RightNoisedIndNb = np.random.randint(1, 11, 1)
        RightNoisedInd = np.sort([2500] + random.sample(range(2500, 3000), RightNoisedIndNb[0])*2 + [3000])
        CandInd = np.reshape(RightNoisedInd, (-1, 2))
        SecDiff = CandInd[:, 1] - CandInd[:, 0]
        RightNoisedInd = CandInd[(SecDiff <= MaxDura) & (SecDiff >= RightMinDura)]
        if len(RightNoisedInd) < 1:
            HFSet[:, range(2500, 3000)] += np.random.normal(0, 0.1, HFSet[:, range(2500, 3000)].shape)
            HFSet[:, range(2500, 3000)] = np.clip(HFSet[:, range(2500, 3000)], 0.01, 0.99)
        else:
            RightNoisedInd = np.concatenate(np.split(np.arange(3000), RightNoisedInd.ravel())[1::2]).copy()
            HFSet[:, RightNoisedInd] += np.random.normal(0, 0.1, HFSet[:, RightNoisedInd].shape)
            HFSet[:, RightNoisedInd] = np.clip(HFSet[:, RightNoisedInd], 0.01, 0.99)
        return HFSet

    # Impulse noise artifact
    def Impulse(self, ImpSet):
        SecSize = 100
        ImpSize = 5
        fs = np.random.uniform(100, 500)
        ImpInd = np.concatenate([np.zeros(600 // SecSize - ImpSize), np.ones(ImpSize)]).astype('bool')
        VecToSat = np.arange(-(SecSize // 2), (SecSize // 2)) + 0.01
        ImpulseArt = 0.02 * np.sin((np.pi * VecToSat) / (fs * np.random.uniform(0.1, 0.4, ImpSize)[:, None])) / (np.pi * VecToSat / fs)
        ImpulseArt = np.random.choice([-1, 1], size=ImpSize, replace=True)[:, None] * ImpulseArt
        ImpulseArt = np.reshape(ImpulseArt, (-1, SecSize * ImpSize)).copy()
        ImpSet[:, -(SecSize * ImpSize):] += ImpulseArt
        ImpSet = np.clip(ImpSet, 0.01, 0.99)
        return ImpSet

    # Low frequency noise artifact
    def LowFreq(self, LFSet):
        partition_idx = np.sort(random.sample(range(3000), np.random.randint(1, 6, 1)[0]))
        ext_partition_idx = np.concatenate([[0], partition_idx, [3000]])
        intv_st_pts = ext_partition_idx[:-1]
        intv_len = ext_partition_idx[1:] - ext_partition_idx[:-1]
        intv_amp = [np.random.uniform(0.05, 0.15)*(x > 600) for x in intv_len]
        intv_coeff = [2*np.pi/x if x > 0 else 1.0 for x in intv_len]
        domain = np.array([])
        image = np.array([])
        for i in range(len(intv_len)):
            domain_tmp = np.arange(ext_partition_idx[i], ext_partition_idx[i+1])
            image_tmp = np.array([1+intv_amp[i]*np.sin(intv_coeff[i]*(x-intv_st_pts[i])) for x in domain_tmp])
            domain = np.concatenate([domain, domain_tmp])
            image = np.concatenate([image, image_tmp])
        LFSet = image * LFSet
        LFSet = np.clip(LFSet, 0.01, 0.99)
        return LFSet

    # Time lag alteration artifact
    def TimeLag(self, TLSet):
        rsp_intv_end_point = np.sort(np.append(np.random.choice(np.arange(1, 2509), 9, replace=False), [0, 2900])) + np.arange(0, 101, 10)
        rsp_intv_len = rsp_intv_end_point[1:] - rsp_intv_end_point[:-1]
        res_diff = [5, -5, 6, -6, 7, -7, 8, -8, 9, -9]
        res_diff = np.random.permutation(res_diff)
        for j in range(10):
            sp = rsp_intv_end_point[j]
            ep = rsp_intv_end_point[j + 1]
            res_tmp = signal.resample(TLSet[:, sp:ep], rsp_intv_len[j] + res_diff[j], axis=-1)
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
    return MSSE

@tf.function
def RMSE(y_true, y_pred):
    SE = (y_true - y_pred)**2
    MSE = tf.reduce_mean(SE)
    return tf.math.sqrt(MSE)
