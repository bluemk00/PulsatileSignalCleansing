import sys
import os
import tensorflow as tf
import numpy as np
import csv
from model import SelfAttentionuNet_1D

# Ensure yaml is installed
try:
    import yaml
except ImportError:
    print("yaml is not installed. Installing now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
    import yaml

# Ensure tqdm is installed
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm is not installed. Installing now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

sys.path.append("../../../lib/")
from functions import *

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

if __name__ == "__main__":

    # User-defined paths
    base_path = config['eval']['paths']['data']
    saved_model_path = config['eval']['paths']['model']

    os.makedirs(config['eval']['paths']['results'], exist_ok=True)
    results_path = config['eval']['paths']['results'] + 'MDP_PPGtoABP_performance.csv'

    # Load data
    GroundTruth = np.load(base_path + 'MIMIC_ABP_highqual.npy')
    Original_LowQual = np.load(base_path + 'MIMIC_PPG_original_lowqual.npy')
    Cleansed_DA = np.load(base_path + 'MIMIC_PPG_cleansed_DA.npy')
    Cleansed_DA_D = np.load(base_path + 'MIMIC_PPG_cleansed_DA_D.npy')
    Cleansed_DA_A = np.load(base_path + 'MIMIC_PPG_cleansed_DA_A.npy')
    Cleansed_HIVAE = np.load(base_path + 'MIMIC_PPG_cleansed_HIVAE.npy')
    Cleansed_GPVAE = np.load(base_path + 'MIMIC_PPG_cleansed_GPVAE.npy')
    Cleansed_SNM = np.load(base_path + 'MIMIC_PPG_cleansed_SNM.npy')
    Cleansed_BDC = np.load(base_path + 'MIMIC_PPG_cleansed_BDC.npy')

    # Clip data
    Original_LowQual = np.clip(Original_LowQual, 0.0, 1.0)
    Cleansed_DA = np.clip(Cleansed_DA, 0.0, 1.0)
    Cleansed_DA_A = np.clip(Cleansed_DA_A, 0.0, 1.0)
    Cleansed_DA_D = np.clip(Cleansed_DA_D, 0.0, 1.0)
    Cleansed_HIVAE = np.clip(Cleansed_HIVAE, 0.0, 1.0)
    Cleansed_GPVAE = np.clip(Cleansed_GPVAE, 0.0, 1.0)
    Cleansed_SNM = np.clip(Cleansed_SNM, 0.0, 1.0)
    Cleansed_BDC = np.clip(Cleansed_BDC, 0.0, 1.0)

    # Load model
    input_shape = (1200, 1)
    model = SelfAttentionuNet_1D(input_shape)
    model.load_weights(saved_model_path)

    # Evaluate models
    results = []
    models = [
        ('Baseline', Original_LowQual),
        ('DA', Cleansed_DA),
        ('DA_D', Cleansed_DA_D),
        ('DA_A', Cleansed_DA_A),
        ('HIVAE', Cleansed_HIVAE),
        ('GPVAE', Cleansed_GPVAE),
        ('SNM', Cleansed_SNM),
        ('BDC', Cleansed_BDC)
    ]

    for model_name, cleansed_data in tqdm(models, desc="Evaluating models"):
        Prediction = model.predict(cleansed_data[:, -1200:])
        Prediction = Prediction.reshape(cleansed_data[:, -1200:].shape)
        Prediction = (Prediction * 200.0) + 20.0
        _, mae, _ = calculate_mse_mae_mre(GroundTruth, Prediction)
        kld_mean, _ = compute_kld_vector(GroundTruth, Prediction, subpart_len=200)
        results.append([model_name, mae, kld_mean])

    # Save results to a CSV file
    with open(results_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'MAE', 'MKLD'])
        writer.writerows(results)