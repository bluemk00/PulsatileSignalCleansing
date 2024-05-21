import numpy as np
import os
import subprocess
import sys

# Ensure tqdm is installed
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm is not installed. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

def main():
    # Get list of .npz files from the directory
    source_dir = './Data_1ModelTrain/MIMIC3_High_Qual_60s/'
    pat_list = [pat for pat in os.listdir(source_dir) if pat.endswith('.npz')]

    # Initialize empty arrays with the correct shape for concatenation
    art_concat = np.empty((0, 6000))
    ppg_concat = np.empty((0, 6000))

    # Load ART and PPG data from each file and concatenate
    for pat in tqdm(pat_list, desc="Loading and concatenating data", unit="file"):
        data = np.load(os.path.join(source_dir, pat), allow_pickle=True)
        art = np.vstack(data['ART'])
        ppg = np.vstack(data['PLETH'])
        
        art_concat = np.vstack((art_concat, art))
        ppg_concat = np.vstack((ppg_concat, ppg))

    # Extract the middle portion of the data (1500:4500)
    art_aggregate = art_concat[:, 1500:4500]
    ppg_aggregate = ppg_concat[:, 1500:4500]

    # Shuffle the ART data and split into training and validation sets
    np.random.shuffle(art_aggregate)
    split_idx = len(art_aggregate) // 5
    TrSet_ART = art_aggregate[split_idx:]
    ValSet_ART = art_aggregate[:split_idx]

    # Ensure the TrainDataSet directory exists
    train_data_dir = './Data_1ModelTrain/MIMIC3_ABP/'
    os.makedirs(train_data_dir, exist_ok=True)

    # Save the ART training and validation sets
    np.save(os.path.join(train_data_dir, 'MIMIC_ART_TrSet.npy'), TrSet_ART)
    np.save(os.path.join(train_data_dir, 'MIMIC_ART_ValSet.npy'), ValSet_ART)

    # Shuffle the PPG data and split into training and validation sets
    np.random.shuffle(ppg_aggregate)
    split_idx = len(ppg_aggregate) // 5
    TrSet_PPG = ppg_aggregate[split_idx:]
    ValSet_PPG = ppg_aggregate[:split_idx]

    # Ensure the TrainDataSet directory exists
    train_data_dir = './Data_1ModelTrain/MIMIC3_PPG/'
    os.makedirs(train_data_dir, exist_ok=True)

    # Save the PPG training and validation sets
    np.save(os.path.join(train_data_dir, 'MIMIC_PPG_TrSet.npy'), TrSet_PPG)
    np.save(os.path.join(train_data_dir, 'MIMIC_PPG_ValSet.npy'), ValSet_PPG)

    # Check the shapes of the resulting datasets
    print(f"TrSet_ART shape: {TrSet_ART.shape}, ValSet_ART shape: {ValSet_ART.shape}")
    print(f"TrSet_PPG shape: {TrSet_PPG.shape}, ValSet_PPG shape: {ValSet_PPG.shape}")

if __name__ == "__main__":
    main()
