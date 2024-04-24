import tensorflow as tf
import numpy as np
import csv
from model import SelfAttentionuNet_1D
from functions import *

# User-defined paths
base_path = '../../../A.Data/A2.Processed/PPG/DownstreamTask/'
saved_model_path = '../results/3090_test1/epoch_919_loss_0.00384_mse_0.00377_mae_0.04514_valloss_0.00488_valmse_0.00447_valmae_0.04959.hdf5'
results_path = './results/MDP_eval.csv'

# Load data
Test_Y = np.load(base_path + 'MIMIC_ABP_TestSet_Y_clean12s_remove_nan.npy')
Uncleaned = np.load(base_path + 'MIMIC_PPG_TestSet_X_clean25s_noise15s_remove_nan.npy')
Cleaned_main = np.load(base_path + 'noisePPG_cleaning_pred_remove_nan/DTW_Pred_PPG_PPGd_AMP_dr1.npy')
Cleaned_ab1 = np.load(base_path + 'noisePPG_cleaning_pred_remove_nan/DTW_Pred_PPG_PPGd_dr1.npy')
Cleaned_ab2 = np.load(base_path + 'noisePPG_cleaning_pred_remove_nan/DTW_Pred_PPG_AMP_dr1.npy')
Cleaned_hivae = np.load(base_path + 'noisePPG_cleaning_pred_remove_nan/DTW_Pred_HIVAE.npy')
Cleaned_gpvae = np.load(base_path + 'noisePPG_cleaning_pred_remove_nan/DTW_Pred_GPVAE.npy')
Cleaned_bdc = np.load(base_path + 'noisePPG_cleaning_pred_remove_nan/DTW_Pred_BDC.npy')
Cleaned_snm = np.load(base_path + 'noisePPG_cleaning_pred_remove_nan/DTW_Pred_SNM.npy')

# Clip data
Uncleaned = np.clip(Uncleaned, 0.0, 1.0)
Cleaned_main = np.clip(Cleaned_main, 0.0, 1.0)
Cleaned_ab1 = np.clip(Cleaned_ab1, 0.0, 1.0)
Cleaned_ab2 = np.clip(Cleaned_ab2, 0.0, 1.0)
Cleaned_hivae = np.clip(Cleaned_hivae, 0.0, 1.0)
Cleaned_gpvae = np.clip(Cleaned_gpvae, 0.0, 1.0)
Cleaned_bdc = np.clip(Cleaned_bdc, 0.0, 1.0)
Cleaned_snm = np.clip(Cleaned_snm, 0.0, 1.0)

# Load model
input_shape = (1200, 1)
model = SelfAttentionuNet_1D(input_shape)
model.load_weights(saved_model_path)

# Evaluate models
results = []
models = [
    ('Uncleaned', Uncleaned),
    ('DA', Cleaned_main),
    ('DA-D', Cleaned_ab2),
    ('DA-A', Cleaned_ab1),
    ('HI-VAE', Cleaned_hivae),
    ('GP-VAE', Cleaned_gpvae),
    ('SNM', Cleaned_snm),
    ('BDC', Cleaned_bdc)
]

for model_name, cleaned_data in models:
    pred = model.predict(cleaned_data[:,-1200:])
    pred = pred.reshape(cleaned_data[:,-1200:].shape)
    pred = (pred*200)+20
    mae, _, _ = calc_mae_mse(Test_Y, pred)
    kld_mean, _ = compute_kld_vector(Test_Y, pred, segment_length=200)
    results.append([model_name, mae, kld_mean])

# Save results to a CSV file
with open(results_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Model', 'MAE', 'MKLD'])
    writer.writerows(results)