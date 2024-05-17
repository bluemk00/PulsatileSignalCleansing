import numpy as np
import time
from dtaidistance import dtw
import tensorflow as tf

def calc_shortest_dtw_optimized_dtaidistance(initial_clean, prediction, sliding=50, max_dist=np.inf):
    """
    Calculate the shortest DTW distance using the 'dtaidistance' library.
    
    Parameters:
    initial_clean (array-like): Initial clean signal.
    prediction (array-like): Prediction signal to compare.
    sliding (int, optional): Sliding window step size. Defaults to 50.
    max_dist (float, optional): Maximum distance threshold for DTW. Defaults to np.inf.
    
    Returns:
    float: Minimum DTW distance between the prediction and the closest matching segment of the initial clean signal.
    
    Raises:
    ValueError: If the length of the initial clean signal is less than the length of the prediction signal.
    """
    if len(initial_clean) < len(prediction):
        raise ValueError("Error: Length of clean signal is less than prediction")

    initial_clean = np.array(initial_clean, dtype=np.double)
    prediction = np.array(prediction, dtype=np.double)

    window_nb = (len(initial_clean) - len(prediction) + sliding) // sliding

    distances = [
        dtw.distance_fast(initial_clean[i * sliding:i * sliding + len(prediction)], prediction, max_dist=max_dist) 
        for i in range(window_nb)
    ]

    return min(distances)

def cummulative_prediction(AEModel, Model_name, noise, org_batch, missing_mask=None, pred_step=100, dtw_time_len=250, att_time_len=250, sliding=50, dtw=True, sqi_return=False, max_dist=20.0):
    """
    Perform a cumulative prediction on a noisy input signal using a specified Autoencoder model and optional DTW-based cleansing.

    Parameters:
    AEModel (tensorflow.python.keras.engine.functional.Functional): Autoencoder model for signal processing.
    Model_name (str): Name of the model.
    noise (array-like): Noisy input data to be processed.
    org_batch (int): Original batch size for processing.
    missing_mask (array-like, optional): Mask for missing data in the input signal. Defaults to None.
    pred_step (int, optional): Prediction sliding interval. Defaults to 100.
    dtw_time_len (int, optional): Time length of the time series for DTW comparison. Defaults to 250.
    att_time_len (int, optional): Time length for replacing the original time series with the predicted one. Defaults to 250.
    sliding (int, optional): Sliding interval for DTW calculation. Defaults to 50.
    dtw (bool, optional): Flag to enable/disable DTW comparison. Defaults to True.
    sqi_return (bool, optional): Flag to return the SQI (Signal Quality Index) vector. Defaults to False.
    max_dist (float, optional): Maximum distance threshold for DTW calculation. Defaults to 20.0.

    Returns:
    array-like: The cumulative output after processing and cleansing. If `sqi_return` is True, also returns the SQI vector.

    Raises:
    ValueError: If the Model_name is not in the specified list of model names.
    """
    
    if Model_name not in ['DI_A','DI_D','DI','DA_A','DA_D','DA','HIVAE','GPVAE','SNM','BDC']:
        raise ValueError("Model name must be in ['DI_A','DI_D','DI','DA_A','DA_D','DA','HIVAE','GPVAE','SNM','BDC'].")

    abp_input = noise.copy()

    if missing_mask is not None:
        mask = missing_mask.copy()
    else:
        if Model_name in ['SNM','BDC']:
            mask = np.zeros_like(abp_input)
            mask[:, :2500] = 1
        else:
            mask = np.ones_like(abp_input)
            mask[:, :2500] = 0

    last_n = abp_input.shape[0] % org_batch
    batch_nb = abp_input.shape[0] // org_batch + 1

    if pred_step < 500:
        add_nan = np.ones_like(abp_input[:, -500 + pred_step:])
        abp_input = np.concatenate((abp_input, add_nan * 0.3), axis=1)
        if Model_name in ['HIVAE','GPVAE']:
            mask = np.concatenate((mask, add_nan), axis=1)
        elif Model_name in ['SNM','BDC']:
            mask = np.concatenate((mask, (1 - add_nan)), axis=1)

    pred_nb = (abp_input.shape[1] - 3000) // pred_step + 1

    start_time = time.time()

    for n in range(batch_nb):
        batch = org_batch
        if n == batch_nb - 1:
            if last_n == 0:
                continue
            else:
                batch = last_n

        initial_clean = abp_input[batch * n:batch * (n + 1), :2500].copy()
        batch_time0 = time.time()

        for i in range(pred_nb):
            if i == 0:
                if n < batch_nb - 1:
                    x = abp_input[batch * n:batch * (n + 1), :3000].copy()
                    if Model_name in ['HIVAE','GPVAE','SNM','BDC']:
                        m = mask[batch * n:batch * (n + 1), :3000].copy()
                else:
                    x = abp_input[-batch:, :3000].copy()
                    if Model_name in ['HIVAE','GPVAE','SNM','BDC']:
                        m = mask[-batch:, :3000].copy()

            if dtw:
                if Model_name in ['DI_A','DI_D','DI','DA_A','DA_D','DA']:
                    Pred = AEModel.predict(x)[0][:, :dtw_time_len]
                elif Model_name in ['HIVAE','GPVAE']:
                    x_resh = np.reshape(x, (x.shape[0], -1, 50))
                    Recon = AEModel.decode(AEModel.encode(x_resh).mean().numpy()).mean().numpy()
                    Recon_resh = np.reshape(Recon, (Recon.shape[0], -1))
                    Impute = Recon_resh * m + x * (1 - m)
                    Pred = Impute[:, 2500:2500 + dtw_time_len]
                elif Model_name == 'SNM':
                    x_resh = np.reshape(x, (x.shape[0], -1, 50))
                    m_resh = np.reshape(m, (m.shape[0], -1, 50))
                    Recon = AEModel.predict([x_resh, m_resh])
                    Recon_resh = np.reshape(Recon, (Recon.shape[0], -1))
                    Pred = Recon_resh[:, 2500:2500 + dtw_time_len]
                else:
                    x_resh = np.reshape(x, (x.shape[0], -1, 1))
                    m_resh = np.reshape(m, (m.shape[0], -1, 1))
                    Recon = AEModel.predict([x_resh, m_resh])
                    Recon_resh = np.reshape(Recon, (Recon.shape[0], -1))
                    Pred = Recon_resh[:, 2500:2500 + dtw_time_len]
                
                Raw = x[:, 2500:2500 + dtw_time_len].copy()

                dtw_time0 = time.time()

                min_dtw_pred = np.array([calc_shortest_dtw_optimized_dtaidistance(initial_clean[ID], Pred[ID], sliding=sliding, max_dist=max_dist) for ID in range(batch)])
                min_dtw_raw = np.array([calc_shortest_dtw_optimized_dtaidistance(initial_clean[ID], Raw[ID], sliding=sliding, max_dist=max_dist) for ID in range(batch)])

                print(f'Time required for DTW calculation ({i + 1}/{pred_nb}): {time.time() - dtw_time0:.5f} sec')
                
                Mask = min_dtw_pred < min_dtw_raw
                if sqi_return:
                    min_dtw = np.where(min_dtw_pred < min_dtw_raw, min_dtw_pred, min_dtw_raw)

                Sel = Pred * Mask[:, None] + Raw * (1 - Mask)[:, None]

                x[:, 2500:2500 + att_time_len] = Sel[:, :att_time_len].copy()
                if Model_name in ['HIVAE','GPVAE']:
                    m[:, 2500:2500 + pred_step] = 0
                elif Model_name in ['SNM','BDC']:
                    m[:, 2500:2500 + pred_step] = 1
                
            else:
                if Model_name in ['DI_A','DI_D','DI','DA_A','DA_D','DA']:
                    Pred = AEModel.predict(x)[0]
                elif Model_name in ['HIVAE','GPVAE']:
                    x_resh = np.reshape(x, (x.shape[0], -1, 50))
                    Recon = AEModel.decode(AEModel.encode(x_resh).mean().numpy()).mean().numpy()
                    Recon_resh = np.reshape(Recon, (Recon.shape[0], -1))
                    Impute = Recon_resh * m + x * (1 - m)
                    Pred = Impute[:, 2500:]
                elif Model_name == 'SNM':
                    x_resh = np.reshape(x, (x.shape[0], -1, 50))
                    m_resh = np.reshape(m, (m.shape[0], -1, 50))
                    Recon = AEModel.predict([x_resh, m_resh])
                    Recon_resh = np.reshape(Recon, (Recon.shape[0], -1))
                    Pred = Recon_resh[:, 2500:]
                else:
                    x_resh = np.reshape(x, (x.shape[0], -1, 1))
                    m_resh = np.reshape(m, (m.shape[0], -1, 1))
                    Recon = AEModel.predict([x_resh, m_resh])
                    Recon_resh = np.reshape(Recon, (Recon.shape[0], -1))
                    Pred = Recon_resh[:, 2500:]
                    
                Sel = Pred[:, :att_time_len].copy()

                x[:, 2500:2500 + att_time_len] = Sel.copy()
                if Model_name in ['HIVAE','GPVAE']:
                    m[:, 2500:2500 + pred_step] = 0
                elif Model_name in ['SNM','BDC']:
                    m[:, 2500:2500 + pred_step] = 1

            if i == 0:
                cumm_output_per_batch = Sel[:, :pred_step].copy()
                if dtw and sqi_return:
                    cumm_sqi_per_batch = min_dtw[:, None]
            else:
                cumm_output_per_batch = np.concatenate((cumm_output_per_batch, Sel[:, :pred_step]), axis=1)
                if dtw and sqi_return:
                    cumm_sqi_per_batch = np.concatenate((cumm_sqi_per_batch, min_dtw[:, None]), axis=1)

            if i < pred_nb - 1:
                if n < batch_nb - 1:
                    new_segment = abp_input[batch * n:batch * (n + 1), 3000 + i * pred_step:3000 + (i + 1) * pred_step].copy()
                    if Model_name in ['HIVAE','GPVAE','SNM','BDC']:
                        new_mask = mask[batch * n:batch * (n + 1), 3000 + i * pred_step:3000 + (i + 1) * pred_step].copy()
                else:
                    new_segment = abp_input[-batch:, 3000 + i * pred_step:3000 + (i + 1) * pred_step].copy()
                    if Model_name in ['HIVAE','GPVAE','SNM','BDC']:
                        new_mask = mask[-batch:, 3000 + i * pred_step:3000 + (i + 1) * pred_step].copy()
                x = np.concatenate((x[:, pred_step:], new_segment), axis=1)
                if Model_name in ['HIVAE','GPVAE','SNM','BDC']:
                    m = np.concatenate((m[:, pred_step:], new_mask), axis=1)

        print(f'Time required for batch: {time.time() - batch_time0:.5f} sec')

        if n == 0:
            cumm_output = cumm_output_per_batch.copy()
            if dtw and sqi_return:
                cumm_sqi = cumm_sqi_per_batch.copy()
        else:
            cumm_output = np.concatenate((cumm_output, cumm_output_per_batch), axis=0)
            if dtw and sqi_return:
                cumm_sqi = np.concatenate((cumm_sqi, cumm_sqi_per_batch), axis=0)
        print(f'Processing {cumm_output.shape[0]} / {abp_input.shape[0]}')

    print(f'{cumm_output.shape[0]} / {abp_input.shape[0]} -- prediction complete')
    print(f'Time required for prediction: {time.time() - start_time:.5f} sec')

    if dtw and sqi_return:
        return cumm_output, cumm_sqi
    else:
        return cumm_output
