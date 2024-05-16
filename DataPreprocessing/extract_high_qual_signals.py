import os
import subprocess
import sys
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Ensure heartpy is installed
try:
    import heartpy as hp
except ImportError:
    print("heartpy is not installed. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "heartpy"])
    import heartpy as hp

# Ensure tqdm is installed
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm is not installed. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

# Ensure sklearn is installed
try:
    from sklearn.preprocessing import minmax_scale
except ImportError:
    print("sklearn is not installed. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sklearn"])
    from sklearn.preprocessing import minmax_scale

# Ensure scipy is installed
try:
    from scipy import signal
except ImportError:
    print("scipy is not installed. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    from scipy import signal

def remove_bad_signal(ppg, abp, ecg, sig_length, fs=125):
    # Calculate the NaN proportion in each signal and BP difference
    TF = np.mean(np.isnan(abp)) + np.mean(np.isnan(ppg)) + np.mean(np.isnan(ecg))
    BP_DIFF = np.max(abp) - np.min(abp)

    # Check initial signal quality
    if (TF > 0) or (BP_DIFF < 20) or (BP_DIFF > 120):
        return 0
    
    # Normalize the signals
    PPG = minmax_scale(ppg)
    ABP = minmax_scale(abp)
    ECG = minmax_scale(ecg)
    
    decision = 1
    
    # Transpose the signals if they are in column format
    if PPG.shape[0] > 1:
        PPG = PPG.T
    if ABP.shape[0] > 1:
        ABP = ABP.T
    if ECG.shape[0] > 1:
        ECG = ECG.T
        
    PPG_r = 1 - PPG
    ABP_r = 1 - ABP
    ECG_r = 1 - ECG

    # Process PPG signal
    try:
        wd_p, m_p = hp.process(PPG, sample_rate=fs)
    except Exception:
        return 0
    pks_PPG = wd_p['peaklist']
    peak_dist_PPG = np.diff(pks_PPG)
    std_peak_dist_PPG = np.std(peak_dist_PPG)
    std_peaks_PPG = np.std(PPG[pks_PPG])
    num_peaks_PPG = len(pks_PPG)

    if (std_peaks_PPG > 0.2) or (std_peak_dist_PPG > 10) or (num_peaks_PPG < int(sig_length / (fs * 2))):
        return 0
        
    if (pks_PPG[0] > 1.5 * np.mean(peak_dist_PPG)) or (sig_length - pks_PPG[-1] > 1.5 * np.mean(peak_dist_PPG)):
        return 0

    # Process ABP signal
    try:
        wd_a, m_a = hp.process(ABP, sample_rate=fs)
    except Exception:
        return 0
    pks_ABP = wd_a['peaklist']
    peak_dist_ABP = np.diff(pks_ABP)
    std_peak_dist_ABP = np.std(peak_dist_ABP)
    std_peaks_ABP = np.std(ABP[pks_ABP])
    num_peaks_ABP = len(pks_ABP)

    if (std_peaks_ABP > 0.15) or (std_peak_dist_ABP > 5) or (num_peaks_ABP < int(sig_length / (fs * 2))):
        return 0
        
    if (pks_ABP[0] > 1.5 * np.mean(peak_dist_ABP)) or (sig_length - pks_ABP[-1] > 1.5 * np.mean(peak_dist_ABP)):
        return 0

    # Process ECG signal
    try:
        wd_e, m_e = hp.process(ECG, sample_rate=fs)
    except Exception:
        return 0
    pks_ECG = [peak for peak in wd_e['peaklist'] if peak not in wd_e['removed_beats']]
    peak_dist_ECG = np.diff(pks_ECG)
    std_peak_dist_ECG = np.std(peak_dist_ECG)
    std_peaks_ECG = np.std(ECG[pks_ECG])
    num_peaks_ECG = len(pks_ECG)
    peak_diff_ECG = np.abs(np.diff(ECG[pks_ECG]))

    # Process reversed ECG signal
    try:
        wd_e_r, m_e_r = hp.process(ECG_r, sample_rate=fs)
    except Exception:
        return 0
    pks_ECG_r = [peak for peak in wd_e_r['peaklist'] if peak not in wd_e_r['removed_beats']]
    peak_dist_ECG_r = np.diff(pks_ECG_r)
    std_peak_dist_ECG_r = np.std(peak_dist_ECG_r)
    std_peaks_ECG_r = np.std(ECG_r[pks_ECG_r])
    num_peaks_ECG_r = len(pks_ECG_r)
    peak_diff_ECG_r = np.abs(np.diff(ECG_r[pks_ECG_r]))

    if ((std_peak_dist_ECG > 20) or (num_peaks_ECG < int(sig_length / (fs * 2)))) and \
       ((std_peak_dist_ECG_r > 20) or (num_peaks_ECG_r < int(sig_length / (fs * 2)))):
        return 0

    if ((pks_ECG[0] > 1.5 * np.mean(peak_dist_ECG)) or (sig_length - pks_ECG[-1] > 1.5 * np.mean(peak_dist_ECG))) and \
       ((pks_ECG_r[0] > 1.5 * np.mean(peak_dist_ECG_r)) or (sig_length - pks_ECG_r[-1] > 1.5 * np.mean(peak_dist_ECG_r))):
        return 0

    if (len(peak_diff_ECG) != 0 and len(peak_diff_ECG_r) != 0):
        if (np.max(peak_diff_ECG) > 0.6) or (np.max(peak_diff_ECG_r) > 0.6):
            return 0
    elif len(peak_diff_ECG_r) != 0 and np.max(peak_diff_ECG_r) > 0.6:
        return 0
    elif len(peak_diff_ECG) != 0 and np.max(peak_diff_ECG) > 0.6:
        return 0
    else:
        return 0
        
    if np.sum(np.abs(np.diff(ECG[:len(ECG) // 2])) < 0.01) < 2:
        return 0

    if np.sum(np.abs(np.diff(ECG[len(ECG) // 2:])) < 0.01) < 2:
        return 0

    return 1

###########################################################################################
fs0 = 125  # Original sampling rate
fs1 = 100  # New sampling rate
time_len = 60  # Seconds
intv = fs0 * time_len  # Segment length (sampling rate * seconds)
ovlp = 0  # Overlapping length between segments
source_dir = 'raw_signals/'
output_dir = f'high_qual_{time_len}s/'
###########################################################################################

if __name__ == "__main__":
    # List all .npz files starting with '3' in the source directory
    rawdata_list = [filename for filename in os.listdir(source_dir) if filename.find('.npz') > 0 and filename.find('3') == 0]
    rawdata_list.sort()

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Folder created.")
    else:
        print("Folder already exists.")
    
    # Process each file in the raw data list
    for p in tqdm(range(len(rawdata_list)), desc="Processing files", unit="file"):
        abp_list = list()
        ppg_list = list()
        ecg_list = list()
        sig_len_list = list()
        record_name_list = list()
        base_time_list = list()

        # Load the raw data
        data = np.load(source_dir + rawdata_list[p], allow_pickle=True)
        seg_nb = len(data['record_name'])  # Number of partial records per one patient

        for i in range(seg_nb):
            L = data['sig_len'][i]
            abp0 = data['ART'][i]
            ppg0 = data['PLETH'][i]
            ecg0 = data['II'][i]

            sub_abp_list = list()
            sub_ppg_list = list()
            sub_ecg_list = list()
            accum_sig_len = 0  # Total length of processed signals per one patient

            # Segment the signals
            for j in range(0, (L - ovlp) // intv):
                abpf = abp0[intv * j:intv * (j + 1) + ovlp]
                ppgf = ppg0[intv * j:intv * (j + 1) + ovlp]
                ecgf = ecg0[intv * j:intv * (j + 1) + ovlp]

                # Check for NaNs in the segment
                if (np.mean(np.isnan(abpf)) > 0) or (np.mean(np.isnan(ppgf)) > 0) or (np.mean(np.isnan(ecgf)) > 0):
                    continue

                # Check for valid ABP and PPG ranges
                if (np.min(abpf) < 30) or (np.max(abpf) > 220) or (np.min(ppgf) < 0):
                    continue

                # Use the custom function to check the quality of the signals
                decision = remove_bad_signal(ppgf, abpf, ecgf, intv + ovlp)
                
                if decision == 1:
                    # Downsample signals to fs1
                    num_samples = int(len(abpf) * fs1 / fs0)
                    abpf_ds = signal.resample(abpf, num_samples)
                    ppgf_ds = signal.resample(ppgf, num_samples)
                    ecgf_ds = signal.resample(ecgf, num_samples)

                    sub_abp_list.append(np.array(abpf_ds))
                    sub_ppg_list.append(np.array(ppgf_ds))
                    sub_ecg_list.append(np.array(ecgf_ds))
                    accum_sig_len += intv * fs1 / fs0

            # Skip if no valid segments found
            if accum_sig_len == 0:
                continue

            # Append the valid segments to the lists
            record_name_list.append(data['record_name'][i])
            sig_len_list.append(int(accum_sig_len))
            base_time_list.append(data['base_time'][i])
            abp_list.append(np.array(sub_abp_list))
            ppg_list.append(np.array(sub_ppg_list))
            ecg_list.append(np.array(sub_ecg_list))

        # Create a dictionary to store the processed data
        wf_dic = {
            'patient': rawdata_list[p][:-4],
            'fs': fs1,
            'record_name': record_name_list,
            'sig_len': sig_len_list,
            'base_time': base_time_list,
            'ART': np.array(abp_list, dtype=object),
            'PLETH': np.array(ppg_list, dtype=object),
            'II': np.array(ecg_list, dtype=object)
        }

        # Save the dictionary as a compressed .npz file if there are valid segments
        if len(wf_dic['sig_len']) != 0:
            np.savez_compressed(output_dir + rawdata_list[p], **wf_dic)
