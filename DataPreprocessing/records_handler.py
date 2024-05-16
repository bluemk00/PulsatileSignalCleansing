import os
import subprocess
import sys
import numpy as np
import pandas as pd

# Ensure wfdb is installed
try:
    import wfdb
except ImportError:
    print("wfdb is not installed. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wfdb"])
    import wfdb

# Ensure tqdm is installed
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm is not installed. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

if __name__ == "__main__":
    # Create an empty DataFrame with specified columns
    df = pd.DataFrame(columns=['patient', 'record', 'sig_len', 'time(s)', 'fs'])

    # Generate patient list (plist)
    plist = [p for p in os.listdir('./physionet.org/files/mimic3wdb/1.0/') if p[0] == '3']

    # Iterate through each patient folder
    for p in tqdm(plist, desc="Processing patients", unit="patient"):
        # Generate a list of patient folders
        patlist = [pat for pat in os.listdir(f'./physionet.org/files/mimic3wdb/1.0/{p}/') if pat[0] == '3']
        
        for pat in tqdm(patlist, desc=f"Processing patient {p}", unit="folder", leave=False):
            # Generate a list of records
            reclist = [rec[:-4] for rec in os.listdir(f'./physionet.org/files/mimic3wdb/1.0/{p}/{pat}/') if rec[-4:] == '.dat' and rec[7] == '_']

            abp = list()
            ppg = list()
            ecg = list()
            sig_len_list = list()
            record_name_list = list()
            base_time_list = list()
            valid_data_found = False  # Flag to check if any valid data is found

            for rec in reclist:
                try:
                    # Read the record data
                    wfdata = wfdb.rdrecord(f'./physionet.org/files/mimic3wdb/1.0/{p}/{pat}/{rec}', channel_names=['ART', 'PLETH', 'II'])

                    # Check if wfdata is valid, contains 3 channels, and has a minimum signal length of 20 seconds
                    if wfdata is not None and len(wfdata.sig_name) == 3 and wfdata.sig_len >= wfdata.fs * 20:
                        valid_data_found = True
                        # Add the data to the DataFrame
                        df.loc[len(df)] = [pat, rec, wfdata.sig_len, round((wfdata.sig_len) / (wfdata.fs), 2), wfdata.fs]

                        # Append data to lists
                        record_name_list.append(wfdata.record_name)
                        sig_len_list.append(wfdata.sig_len)
                        base_time_list.append(wfdata.base_time)
                        abp.append(wfdata.p_signal[:, 0])
                        ppg.append(wfdata.p_signal[:, 1])
                        ecg.append(wfdata.p_signal[:, 2])
                except Exception as e:
                    # Print the error message if an exception occurs
                    print(f"An error occurred while processing {rec}: {e}")
                    # Pass if any error occurs
                    pass

            if valid_data_found:
                # Create a dictionary to store the patient's data
                wfdb_dic = {
                    'patient': str(pat),
                    'fs': wfdata.fs if 'wfdata' in locals() else None,  # fs from the last successful read
                    'record_name': record_name_list,
                    'sig_len': sig_len_list,
                    'base_time': base_time_list,
                    'ART': np.array(abp),
                    'PLETH': np.array(ppg),
                    'II': np.array(ecg)
                }

                # Ensure the 'raw_signals' directory exists
                if not os.path.exists('raw_signals'):
                    os.makedirs('raw_signals')

                # Save the dictionary as a compressed .npz file
                np.savez_compressed(f'raw_signals/{pat}.npz', **wfdb_dic)

    # Save the DataFrame to a CSV file
    df.to_csv('records_list.csv', index=False)
