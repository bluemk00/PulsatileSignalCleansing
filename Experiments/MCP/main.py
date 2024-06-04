import sys
import os
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm

# Append necessary paths
sys.path.append('../../lib/')

# Import custom functions and models
from functions import *

batch = 1000

if __name__ == "__main__":

    # # External Validation for ABP Cleansing
    # artifact_list = ['high_qual', 'satmax', 'satmin', 'reduced', 'highfreq', 'impulse', 'incomplete']
    # model_list = ['DI', 'DI_D', 'DI_A', 'HIVAE', 'GPVAE', 'SNM', 'BDC']

    # file_path = './ProcessedData/VitalDB_ABP/'
    # true = np.load(file_path + 'Original/VitalDB_ABP_high_qual.npy')
    # target = true[:, -500:]

    # result_for_ABP = pd.DataFrame(columns=['Model Type', 'Artifact Type', 'MSE', 'MAE', 'MRE', 'MKLD'])

    # # Evaluation loop
    # for artifact_type, model_name in tqdm(itertools.product(artifact_list, model_list), total=len(artifact_list) * len(model_list), desc="Evaluating models for ABP"):
    #     pred = np.load(file_path + f'{model_name}_pred/Pred_{model_name}_{artifact_type}.npy')

    #     mse, mae, mre = calculate_mse_mae_mre(target * 200.0 + 20.0, pred * 200.0 + 20.0)
    #     mkld, _ = compute_kld_vector(target * 200.0 + 20.0, pred * 200.0 + 20.0, subpart_len=500)
    #     result_for_ABP.loc[len(result_for_ABP)] = [model_name, artifact_type, mse, mae, mre, mkld]

    # # Save results to CSV
    # result_path = './Results/'
    # os.makedirs(result_path, exist_ok=True)
    # result_for_ABP.to_csv(result_path + 'MCP_for_ABP_Cleansing.csv', index=False)

    # # Save mean metrics for each model
    # result_for_ABP['Model Type'] = pd.Categorical(result_for_ABP['Model Type'], categories=model_list, ordered=True)
    # result_for_ABP_mean = result_for_ABP.groupby('Model Type').mean()
    # result_for_ABP_mean.to_csv(result_path + 'MCP_for_ABP_Cleansing_mean.csv')


    # External Validation for PPG Cleansing
    artifact_list = ['high_qual', 'satmax', 'satmin', 'reduced', 'highfreq', 'impulse', 'incomplete']
    model_list = ['DA', 'DA_D', 'DA_A', 'HIVAE', 'GPVAE', 'SNM', 'BDC']

    file_path = './ProcessedData/VitalDB_PPG/'
    true = np.load(file_path + 'Original/VitalDB_PPG_high_qual.npy')
    target = true[:, -500:]

    result_for_PPG = pd.DataFrame(columns=['Model Type', 'Artifact Type', 'MSE', 'MAE', 'MRE', 'MKLD'])

    # Evaluation loop
    for artifact_type, model_name in tqdm(itertools.product(artifact_list, model_list), total=len(artifact_list) * len(model_list), desc="Evaluating models for PPG"):
        pred = np.load(file_path + f'{model_name}_pred/Pred_{model_name}_{artifact_type}.npy')

        mse, mae, mre = calculate_mse_mae_mre(target * 100.0, pred * 100.0)
        mkld, _ = compute_kld_vector(target * 100.0, pred * 100.0, subpart_len=500)
        result_for_PPG.loc[len(result_for_PPG)] = [model_name, artifact_type, mse, mae, mre, mkld]

    # Evaluation loop
    for artifact_type in artifact_list:
        for model_name in model_list:
            pred = np.load(file_path + f'{model_name}_pred/Pred_{model_name}_{artifact_type}.npy')

        mse, mae, mre = calculate_mse_mae_mre(target * 100.0, pred * 100.0)
        mkld, _ = compute_kld_vector(target * 100.0, pred * 100.0, subpart_len=500)
        result_for_PPG.loc[len(result_for_PPG)] = [model_name, artifact_type, mse, mae, mre, mkld]

    # Save results to CSV
    result_path = './Results/'
    os.makedirs(result_path, exist_ok=True)
    result_for_PPG.to_csv(result_path + 'MCP_for_PPG_Cleansing_1.csv', index=False)

    # Save mean metrics for each model
    result_for_PPG['Model Type'] = pd.Categorical(result_for_PPG['Model Type'], categories=model_list, ordered=True)
    result_for_PPG_mean = result_for_PPG.groupby('Model Type').mean()
    result_for_PPG_mean.to_csv(result_path + 'MCP_for_PPG_Cleansing_mean_1.csv')
