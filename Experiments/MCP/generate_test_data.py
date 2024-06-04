import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D
from tensorflow.keras import Model

# Ensure tqdm is installed
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm is not installed. Installing now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

# Ensure dtaidistance is installed
try:
    from dtaidistance import dtw
except ImportError:
    print("dtaidistance is not installed. Installing now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "dtaidistance"])
    from dtaidistance import dtw

# Configure GPU settings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.98
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))     
tf.compat.v1.enable_eager_execution()

# Append necessary paths
sys.path.append('../../lib/')
sys.path.append('../../Models/utils/')

# Import custom functions and models
from LoadModels import *
from functions import *
from cumm_pred_dtaidtw import *
from GPVAE import *
from BDC_utils import *
from SNM_GRUD import *
from SNM_Interpolate import *
from SNM_SupNotMIWAE import *
from MAIN_ModelStructure import build_model_structure

batch = 1000

if __name__ == "__main__":

    # External Validation for ABP Cleansing
    ABP_model_path = '../../Models/BestModels/ABP/'
    DI, DI_D, DI_A, HIVAE, GPVAE, SNM, BDC = load_models_ABP(ABP_model_path)

    artifact_list = ['high_qual', 'satmax', 'satmin', 'reduced', 'highfreq', 'impulse', 'incomplete']
    model_list = ['DI', 'DI_D', 'DI_A', 'HIVAE', 'GPVAE', 'SNM', 'BDC']
    cleansing_models = [DI, DI_D, DI_A, HIVAE, GPVAE, SNM, BDC]

    file_path = './ProcessedData/VitalDB_ABP/'
    true = np.load(file_path + 'Original/VitalDB_ABP_high_qual.npy')
    target = true[:, -500:]
    mask0 = np.load(file_path + 'Original/VitalDB_ABP_mask0.npy')
    mask1 = np.load(file_path + 'Original/VitalDB_ABP_mask1.npy')

    result_for_ABP = pd.DataFrame(columns=['Model Type', 'Artifact Type', 'MSE', 'MAE', 'MRE', 'MKLD'])

    # Evaluation loop
    for i in range(len(artifact_list)):
        artifact_type = artifact_list[i]
        low_qual = np.load(file_path + f'Original/VitalDB_ABP_{artifact_type}.npy')
        for m in range(len(model_list)):
            model_name = model_list[m]
            Model = cleansing_models[m]
            print(f'Starting the cleansing process with the {model_name} model for {artifact_type} type data.')
            if model_name in ['DI', 'DI_D', 'DI_A', 'DA', 'DA_D', 'DA_A']:
                pred = cummulative_prediction(Model, model_name, low_qual, org_batch=batch, pred_step=500, att_time_len=500, dtw=False)
            elif model_name in ['HIVAE', 'GPVAE']:
                pred = cummulative_prediction(Model, model_name, low_qual, org_batch=batch, missing_mask=mask1, pred_step=500, att_time_len=500, dtw=False)
            elif model_name in ['SNM', 'BDC']:
                pred = cummulative_prediction(Model, model_name, low_qual, org_batch=batch, missing_mask=mask0, pred_step=500, att_time_len=500, dtw=False)
            else:
                continue

            mse, mae, mre = calculate_mse_mae_mre(target * 200.0 + 20.0, pred * 200.0 + 20.0)
            mkld, _ = compute_kld_vector(target * 200.0 + 20.0, pred * 200.0 + 20.0, subpart_len=500)
            result_for_ABP.loc[len(result_for_ABP)] = [model_name, artifact_type, mse, mae, mre, mkld]

            # Uncomment to save the prediction outcomes.
            save_path = file_path + f'{model_name}_pred/'
            os.makedirs(save_path, exist_ok=True)
            np.save(save_path + f'Pred_{model_name}_{artifact_type}.npy', pred)

    # Save results to CSV
    result_path = './Results/'
    os.makedirs(result_path, exist_ok=True)
    result_for_ABP.to_csv(result_path + 'MCP_for_ABP_Cleansing.csv', index=False)

    # Save mean metrics for each model
    result_path = './Results/'
    result_for_ABP['Model Type'] = pd.Categorical(result_for_ABP['Model Type'], categories=model_list, ordered=True)
    result_for_ABP_mean = result_for_ABP.groupby('Model Type').mean()
    result_for_ABP_mean.to_csv(result_path + 'MCP_for_ABP_Cleansing_mean.csv')

    # External Validation for PPG Cleansing
    PPG_model_path = '../../Models/BestModels/PPG/'
    DA, DA_D, DA_A, HIVAE, GPVAE, SNM, BDC = load_models_PPG(PPG_model_path)

    artifact_list = ['high_qual', 'satmax', 'satmin', 'reduced', 'highfreq', 'impulse', 'incomplete']
    model_list = ['DA', 'DA_D', 'DA_A', 'HIVAE', 'GPVAE', 'SNM', 'BDC']
    cleansing_models = [DA, DA_D, DA_A, HIVAE, GPVAE, SNM, BDC]

    file_path = './ProcessedData/VitalDB_PPG/'
    true = np.load(file_path + 'Original/VitalDB_PPG_high_qual.npy')
    target = true[:, -500:]
    mask0 = np.load(file_path + 'Original/VitalDB_PPG_mask0.npy')
    mask1 = np.load(file_path + 'Original/VitalDB_PPG_mask1.npy')

    result_for_PPG = pd.DataFrame(columns=['Model Type', 'Artifact Type', 'MSE', 'MAE', 'MRE', 'MKLD'])

    # Evaluation loop
    for i in range(len(artifact_list)):
        artifact_type = artifact_list[i]
        low_qual = np.load(file_path + f'Original/VitalDB_PPG_{artifact_type}.npy')
        for m in range(len(model_list)):
            model_name = model_list[m]
            Model = cleansing_models[m]
            print(f'Starting the cleansing process with the {model_name} model for {artifact_type} type data.')
            if model_name in ['DI', 'DI_D', 'DI_A', 'DA', 'DA_D', 'DA_A']:
                pred = cummulative_prediction(Model, model_name, low_qual, org_batch=batch, pred_step=500, att_time_len=500, dtw=False)
            elif model_name in ['HIVAE', 'GPVAE']:
                pred = cummulative_prediction(Model, model_name, low_qual, org_batch=batch, missing_mask=mask1, pred_step=500, att_time_len=500, dtw=False)
            elif model_name in ['SNM', 'BDC']:
                pred = cummulative_prediction(Model, model_name, low_qual, org_batch=batch, missing_mask=mask0, pred_step=500, att_time_len=500, dtw=False)
            else:
                continue

            mse, mae, mre = calculate_mse_mae_mre(target * 100.0, pred * 100.0)
            mkld, _ = compute_kld_vector(target * 100.0, pred * 100.0, subpart_len=500)
            result_for_PPG.loc[len(result_for_PPG)] = [model_name, artifact_type, mse, mae, mre, mkld]

            # Uncomment to save the prediction outcomes.
            save_path = file_path + f'{model_name}_pred/'
            os.makedirs(save_path, exist_ok=True)
            np.save(save_path + f'Pred_{model_name}_{artifact_type}.npy', pred)

    # Save results to CSV
    result_for_PPG.to_csv(result_path + 'MCP_for_PPG_Cleansing.csv', index=False)

    # Save mean metrics for each model
    result_for_PPG['Model Type'] = pd.Categorical(result_for_PPG['Model Type'], categories=model_list, ordered=True)
    result_for_PPG_mean = result_for_PPG.groupby('Model Type').mean()
    result_for_PPG_mean.to_csv(result_path + 'MCP_for_PPG_Cleansing_mean.csv')
