import numpy as np
from scipy.special import kl_div

def calc_mae_mse(arr1, arr2):
    mae = np.mean(np.abs(arr1-arr2))
    mse = np.mean((arr1-arr2)**2)
    mre = np.mean(np.abs((arr1-arr2)/(arr1+1e-10)))
    return mae, mse, mre
    
def compute_esd(signal):
    ft = np.fft.fft(signal)
    esd = np.abs(ft)**2
    
    return esd

def compute_kld(sig1, sig2):

    if len(sig1) != len(sig2):
        return "Length of sig1 and sig2 should be the same"

    esd1 = compute_esd(sig1)
    esd2 = compute_esd(sig2)
    
    # ESD를 확률분포로 정규화
    esd1_normalized = esd1 / np.sum(esd1)
    esd2_normalized = esd2 / np.sum(esd2)
    
    # KLD 계산
    kld = np.sum(kl_div(esd1_normalized, esd2_normalized))

    return kld

def compare_partial_kld(sig1, sig2, segment_length=500):

    min_len = min(len(sig1), len(sig2))

    avg_kld = 0.0
    max_kld = 0.0
    seg_range = (min_len)//segment_length
    for i in range(seg_range):
        kld = compute_kld(sig1[i*segment_length:(i+1)*segment_length], sig2[i*segment_length:(i+1)*segment_length])
        avg_kld += kld
        if kld > max_kld:
            max_kld = kld
    avg_kld /= seg_range

    return avg_kld, max_kld

def compute_kld_vector(signals, target_signals, segment_length=500):

    if len(signals) != len(target_signals):
        print("Error: The lengths of the two lists are not equal.")
        return

    kld_vector = [compare_partial_kld(signals[i], target_signals[i], segment_length=segment_length) for i in range(len(signals))]
    kld_vector = np.array(kld_vector)
    kld_mean = np.mean(kld_vector[:,0])
    kld_max = np.mean(kld_vector[:,1])

    return kld_mean, kld_max