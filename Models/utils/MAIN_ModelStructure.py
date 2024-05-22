import os
import sys
sys.path.append("../../lib/")
from artifact_augmentation import RMSE, AmplitudeCost
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, Permute, Reshape, LayerNormalization, LSTM, Bidirectional, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class ModelStructure:
    def __init__(self, modeltype, outptype, gn=0.0, dr=0.0, frame_size=50, overlap_size=10, input_shape=(3000,), out_sec=5, fs=100):
        """
        Initialize the ModelStructure class.
        
        Parameters:
        modeltype (str): Type of model ('DI' or 'DA').
        outptype (int): Type of output (0, 1, or 2).
        gn (float): Gaussian noise level.
        dr (float): Dropout rate.
        frame_size (int): Size of the frames.
        overlap_size (int): Size of the overlap between frames.
        input_shape (tuple): Shape of the input layer.
        out_sec (int): Output seconds.
        fs (int): Sampling frequency.
        """
        if modeltype not in ['DI', 'DA']:
            raise ValueError("Invalid modeltype value. Must be 'DI' or 'DA'.")
        if outptype not in [0, 1, 2]:
            raise ValueError("Invalid outptype value. Must be 0, 1, or 2.")
        self.modeltype = modeltype
        self.outptype = outptype
        self.gn = gn
        self.dr = dr
        self.frame_size = frame_size
        self.overlap_size = overlap_size
        self.input_shape = input_shape
        self.out_sec = out_sec
        self.fs = fs

    def get_save_paths(self):
        """
        Get the save folder and file path based on the output type.
        
        Returns:
        SaveFolder (str): Folder path to save model checkpoints.
        SaveFilePath (str): File path template for model checkpoints.
        """
        if self.outptype == 0:
            if self.modeltype == 'DI':
                SaveFolder = f'./ModelResults/ABPCleansing/{self.modeltype}_A/'
            else:
                SaveFolder = f'./ModelResults/PPGCleansing/{self.modeltype}_A/'
            SaveFilePath = f'{self.modeltype}_A_{{epoch:04}}_val{{val_loss:.7f}}_valOut{{val_Output_loss:.7f}}_valDiff{{val_Diff_loss:.7f}}_loss{{loss:.7f}}_Out{{Output_loss:.7f}}_Diff{{Diff_loss:.7f}}.hdf5'
        elif self.outptype == 1:
            if self.modeltype == 'DI':
                SaveFolder = f'./ModelResults/ABPCleansing/{self.modeltype}_D/'
            else:
                SaveFolder = f'./ModelResults/PPGCleansing/{self.modeltype}_D/'
            SaveFilePath = f'{self.modeltype}_D_{{epoch:04}}_val{{val_loss:.7f}}_valOut{{val_Output_loss:.7f}}_valAmp{{val_Amp_loss:.7f}}_loss{{loss:.7f}}_Out{{Output_loss:.7f}}_Amp{{Amp_loss:.7f}}.hdf5'
        elif self.outptype == 2:
            if self.modeltype == 'DI':
                SaveFolder = f'./ModelResults/ABPCleansing/{self.modeltype}/'
            else:
                SaveFolder = f'./ModelResults/PPGCleansing/{self.modeltype}/'
            SaveFilePath = f'{self.modeltype}_{{epoch:04}}_val{{val_loss:.7f}}_valOut{{val_Output_loss:.7f}}_valDiff{{val_Diff_loss:.7f}}_valAmp{{val_Amp_loss:.7f}}_loss{{loss:.7f}}_Out{{Output_loss:.7f}}_Diff{{Diff_loss:.7f}}_Amp{{Amp_loss:.7f}}.hdf5'
        else:
            raise ValueError("Invalid outptype value. Must be 0, 1, or 2.")

        # Check if the SaveFolder exists, if not, create it
        os.makedirs(SaveFolder, exist_ok=True)
        
        return SaveFolder, SaveFilePath

    def get_optimizer(self):
        """
        Get the Adam optimizer with specific learning rate and decay.
        
        Returns:
        optimizer (Adam): Adam optimizer instance.
        """
        lrate = 0.0005
        decay = 1e-6
        return Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)

    def get_loss_functions(self):
        """
        Get the loss functions based on the output type.
        
        Returns:
        loss_set (dict): Dictionary of loss functions for the model outputs.
        """
        if self.outptype == 0:
            loss_set = {'Output': 'mse', 'Diff': RMSE}
        elif self.outptype == 1:
            loss_set = {'Output': 'mse', 'Amp': AmplitudeCost}
        elif self.outptype == 2:
            loss_set = {'Output': 'mse', 'Diff': RMSE, 'Amp': AmplitudeCost}
        else:
            raise ValueError("Invalid outptype value. Must be 0, 1, or 2.")
        
        return loss_set

    def build_model(self):
        """
        Build a deep learning model for signal processing with different output types.
        
        Returns:
        AEModel (Model): Compiled Keras model.
        SaveFolder (str): Folder path to save model checkpoints.
        SaveFilePath (str): File path template for model checkpoints.
        """
        AEModel = build_model_structure(self.outptype, self.gn, self.dr, self.frame_size, self.overlap_size, self.input_shape, self.out_sec, self.fs)

        # Get optimizer
        optimizer = self.get_optimizer()

        # Compile the model with Adam optimizer
        AEModel.compile(loss=self.get_loss_functions(), optimizer=optimizer)

        # Get save folder and file path
        SaveFolder, SaveFilePath = self.get_save_paths()

        return AEModel, SaveFolder, SaveFilePath


def build_model_structure(outptype, gn=0.0, dr=0.0, frame_size=50, overlap_size=10, input_shape=(3000,), out_sec=5, fs=100):
    """
    Define and build the model structure.
    
    Parameters:
    outptype (int): Determines the type of output and loss function to use.
    gn (float): Gaussian noise level.
    dr (float): Dropout rate.
    frame_size (int): Size of the frames.
    overlap_size (int): Size of the overlap between frames.
    input_shape (tuple): Shape of the input layer.
    out_sec (int): Output seconds.
    fs (int): Sampling frequency.
    
    Returns:
    model (Model): Uncompiled Keras model.
    """
    # Define input layer
    InpL = Input(shape=input_shape)

    # Frame the input signal
    InpFrame = tf.signal.frame(InpL, frame_size, overlap_size)
    InpFrameNoise = GaussianNoise(gn)(InpFrame)

    # Encoder layers
    Encoder = Dense(frame_size, activation='relu', name='Encoder1')(InpFrameNoise)
    Encoder = Dropout(dr)(Encoder, training=True)
    Encoder = Bidirectional(LSTM(int(frame_size / 2), return_sequences=True))(Encoder)

    # Attention mechanism (front)
    Att_front = Bidirectional(LSTM(out_sec, return_sequences=True))(Encoder)
    Att_front = LayerNormalization(axis=1, epsilon=0.001)(Att_front)
    Att_front = Permute((2, 1))(Att_front)
    Att_front = Dropout(dr)(Att_front, training=True)
    Att_front = Dense(InpFrame.shape[1], activation='softmax')(Att_front)
    Att_front = Permute((2, 1), name='Att_front')(Att_front)

    # Context computation
    Context = InpFrameNoise[:, :, None] * Att_front[:, :, :, None]
    Context = tf.reduce_sum(Context, axis=1, name='Context')

    # Decoder layers
    Decoder = Bidirectional(LSTM(int(frame_size / 2), return_sequences=True))(Context)
    Decoder = LayerNormalization(axis=(1, 2), epsilon=0.001)(Decoder)

    # Attention mechanism (back)
    Att_back = Bidirectional(LSTM(frame_size * out_sec, return_sequences=False))(InpFrame)
    Att_back = Reshape((int(fs / frame_size) * out_sec, frame_size))(Att_back)
    Att_back = LayerNormalization(axis=(1, 2), epsilon=0.001)(Att_back)
    Att_back = Bidirectional(LSTM(int(frame_size / 2), return_sequences=True))(Att_back)
    Att_back = Dropout(dr)(Att_back, training=True)
    Att_back = Dense(frame_size, activation='tanh')(Att_back)

    # Scaling and final layers
    Scaling = Decoder + Att_back
    Scaling = Bidirectional(LSTM(int(frame_size / 2), return_sequences=True, name='Scaling'))(Scaling)
    ValDecoder = Context + Scaling
    Output = Reshape((-1,), name='Output')(ValDecoder)
    Diff = Reshape((-1,), name='Diff')(Output[:, 1:] - Output[:, :-1])

    # Define different outputs based on the outptype parameter
    if outptype in [1, 2]:
        Casted = tf.cast(Output, tf.complex64)
        fft = tf.signal.fft(Casted)[:, :(Casted.shape[-1] // 2 + 1)]
        Amp = tf.abs(fft[:, 1:]) * (1 / fft.shape[-1])
        Amp = Reshape((-1,), name='Amp')(Amp)

    if outptype == 0:
        model = Model(InpL, [Output, Diff])
    elif outptype == 1:
        model = Model(InpL, [Output, Amp])
    elif outptype == 2:
        model = Model(InpL, [Output, Diff, Amp])
    else:
        raise ValueError("Invalid outptype value. Must be 0, 1, or 2.")

    return model
