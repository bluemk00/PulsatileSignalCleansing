import numpy as np
import os

# Get list of .npz files from the directory
pat_list = [pat for pat in os.listdir('high_qual_60s/') if pat.endswith('.npz')]

# Initialize empty arrays with the correct shape for concatenation
art_concat = np.empty((0, 6000))
ppg_concat = np.empty((0, 6000))

# Load ART and PPG data from each file and concatenate
for pat in pat_list:
    data = np.load(f'high_qual_60s/{pat}', allow_pickle=True)
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
train_data_dir = '../Models/TrainDataSet/MIMIC_ABP/'
if not os.path.exists(train_data_dir):
    os.makedirs(train_data_dir)
    print("TrainDataSet directory created.")
else:
    print("TrainDataSet directory already exists.")

# Save the ART training and validation sets
np.save(os.path.join(train_data_dir, 'MIMIC_ART_TrSet.npy'), TrSet_ART)
np.save(os.path.join(train_data_dir, 'MIMIC_ART_ValSet.npy'), ValSet_ART)

# Shuffle the PPG data and split into training and validation sets
np.random.shuffle(ppg_aggregate)
split_idx = len(ppg_aggregate) // 5
TrSet_PPG = ppg_aggregate[split_idx:]
ValSet_PPG = ppg_aggregate[:split_idx]

# Ensure the TrainDataSet directory exists
train_data_dir = '../Models/TrainDataSet/MIMIC_PPG/'
if not os.path.exists(train_data_dir):
    os.makedirs(train_data_dir)
    print("TrainDataSet directory created.")
else:
    print("TrainDataSet directory already exists.")

# Save the PPG training and validation sets
np.save(os.path.join(train_data_dir, 'MIMIC_PPG_TrSet.npy'), TrSet_PPG)
np.save(os.path.join(train_data_dir, 'MIMIC_PPG_ValSet.npy'), ValSet_PPG)

# Check the shapes of the resulting datasets
print(f"TrSet_ART shape: {TrSet_ART.shape}, ValSet_ART shape: {ValSet_ART.shape}")
print(f"TrSet_PPG shape: {TrSet_PPG.shape}, ValSet_PPG shape: {ValSet_PPG.shape}")
