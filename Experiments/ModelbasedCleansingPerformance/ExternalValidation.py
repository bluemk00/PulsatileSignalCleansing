import numpy as np
import pandas as pd
import sys
sys.path.append('../../Models/LoadModelScripts/')
sys.path.append('../../lib/')
from LoadModels import load_models
from functions import *
from cumm_pred_dtaidtw import *

def main():
    # Load models
    DI, DI_D, DI_A, HIVAE, GPVAE, SNM, BDC = load_models()

    batch = 1000

    artifact_list = ['high_qual', 'satmax', 'satmin', 'reduced', 'highfreq', 'impulse', 'incomplete']
    model_list = ['DI', 'DI_D', 'DI_A', 'HIVAE', 'GPVAE', 'SNM', 'BDC']
    cleansing_models = [DI, DI_D, DI_A, HIVAE, GPVAE, SNM, BDC]

    file_path = './TestDataSet/ABP/'
    true = np.load(file_path + 'Original/VitalDB_ABP_high_qual.npy')
    target = true[:, -500:]
    mask0 = np.load(file_path + 'Original/VitalDB_ABP_mask0.npy')
    mask1 = np.load(file_path + 'Original/VitalDB_ABP_mask1.npy')

    df = pd.DataFrame(columns=['Model Type', 'Artifact Type', 'MSE', 'MAE', 'MRE', 'MKLD'])

    # Evaluation loop
    for i in range(len(artifact_list)):
        artifact_type = artifact_list[i]
        low_qual = np.load(file_path + f'Original/VitalDB_ABP_{artifact_type}.npy')
        for m in range(len(model_list)):
            model_name = model_list[m]
            Model = cleansing_models[m]
            if model_name in ['DI', 'DI_D', 'DI_A', 'DA', 'DA_D', 'DA_A']:
                pred = cummulative_prediction(Model, model_name, low_qual, org_batch=batch, pred_step=500, att_time_len=500, dtw=False)
                np.save(f'./TestDataSet/ABP/{model_name}_pred/Pred_{model_name}_{artifact_type}.npy', pred)
                mse, mae, mre = calculate_mse_mae_mre(target * 200.0 + 20.0, pred * 200.0 + 20.0)
                mkld, _ = compute_kld_vector(target * 200.0 + 20.0, pred * 200.0 + 20.0, subpart_len=500)
                df.loc[len(df)] = [model_name, artifact_type, mse, mae, mre, mkld]
            elif model_name in ['HIVAE', 'GPVAE']:
                pred = cummulative_prediction(Model, model_name, low_qual, org_batch=batch, missing_mask=mask1, pred_step=500, att_time_len=500, dtw=False)
                np.save(f'./TestDataSet/ABP/{model_name}_pred/Pred_{model_name}_{artifact_type}.npy', pred)
                mse, mae, mre = calculate_mse_mae_mre(target * 200.0 + 20.0, pred * 200.0 + 20.0)
                mkld, _ = compute_kld_vector(target * 200.0 + 20.0, pred * 200.0 + 20.0, subpart_len=500)
                df.loc[len(df)] = [model_name, artifact_type, mse, mae, mre, mkld]
            elif model_name in ['SNM', 'BDC']:
                pred = cummulative_prediction(Model, model_name, low_qual, org_batch=batch, missing_mask=mask0, pred_step=500, att_time_len=500, dtw=False)
                np.save(f'./TestDataSet/ABP/{model_name}_pred/Pred_{model_name}_{artifact_type}.npy', pred)
                mse, mae, mre = calculate_mse_mae_mre(target * 200.0 + 20.0, pred * 200.0 + 20.0)
                mkld, _ = compute_kld_vector(target * 200.0 + 20.0, pred * 200.0 + 20.0, subpart_len=500)
                df.loc[len(df)] = [model_name, artifact_type, mse, mae, mre, mkld]
            else:
                continue
            print(f'[{model_name}, {artifact_type}, {mse}, {mae}, {mre}, {mkld}]')
            # Save results to CSV
            df.to_csv('./Results/External_Validation_for_ABP_Cleansing.csv', index=False)
        

if __name__ == "__main__":
    main()
