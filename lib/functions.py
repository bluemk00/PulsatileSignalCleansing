import numpy as np
from scipy.special import kl_div


def calculate_mse_mae_mre(true_set, pred_set):
    # Ensure both sets have the same number of signals
    if len(true_set) != len(pred_set):
        return "Number of signals in true_set and pred_set should be the same"
    
    total_mse, total_mae, total_mre = 0, 0, 0
    epsilon = 1e-10  # Small constant to avoid division by zero
    
    for true_sig, pred_sig in zip(true_set, pred_set):
        # Ensure both signals have the same length
        if len(true_sig) != len(pred_sig):
            return "Length of each signal in true_set and pred_set should be the same"
        
        # Convert to numpy arrays
        true_sig, pred_sig = np.array(true_sig), np.array(pred_sig)

        # Calculate MSE, MAE, MRE
        mse = np.mean((true_sig - pred_sig) ** 2)
        mae = np.mean(np.abs(true_sig - pred_sig))
        mre = np.mean(np.abs((true_sig - pred_sig) / (true_sig + epsilon)))
        
        total_mse += mse
        total_mae += mae
        total_mre += mre
    
    # Calculate averages
    n = len(true_set)
    avg_mse = total_mse / n
    avg_mae = total_mae / n
    avg_mre = total_mre / n
    
    return avg_mse, avg_mae, avg_mre


def compute_esd(signal):
    # Compute Energy Spectral Density (ESD) using FFT
    ft = np.fft.fft(signal)
    esd = np.abs(ft)**2
    return esd

def compute_kld(true_sig, pred_sig):
    # Check if signal lengths are equal
    if len(true_sig) != len(pred_sig):
        return "Length of true_sig and pred_sig should be the same"

    # Compute ESD for both signals
    esd_true = compute_esd(true_sig)
    esd_pred = compute_esd(pred_sig)
    
    # Normalize ESD to probability distribution
    esd_true_norm = esd_true / np.sum(esd_true)
    esd_pred_norm = esd_pred / np.sum(esd_pred)
    
    # Compute KLD
    kld = np.sum(kl_div(esd_true_norm, esd_pred_norm))

    return kld

def compare_partial_kld(true_sig, pred_sig, subpart_len=500):
    min_len = min(len(true_sig), len(pred_sig))
    avg_kld = 0.0
    max_kld = 0.0
    seg_range = min_len // subpart_len

    if seg_range == 0:  # if sub-part length is greater than signal length
        return avg_kld, max_kld

    for i in range(seg_range):
        kld = compute_kld(true_sig[i*subpart_len:(i+1)*subpart_len], pred_sig[i*subpart_len:(i+1)*subpart_len])
        avg_kld += kld
        if kld > max_kld:
            max_kld = kld
    avg_kld /= seg_range

    return avg_kld, max_kld

def compute_kld_vector(true_set, pred_set, subpart_len=500):
    if len(true_set) != len(pred_set):
        print("Error: The lengths of the two lists are not equal.")
        return

    kld_vector = [compare_partial_kld(true_set[i], pred_set[i], subpart_len) for i in range(len(true_set))]
    kld_vector = np.array(kld_vector)
    kld_mean = np.mean(kld_vector[:, 0])
    kld_max = np.mean(kld_vector[:, 1])

    return kld_mean, kld_max
