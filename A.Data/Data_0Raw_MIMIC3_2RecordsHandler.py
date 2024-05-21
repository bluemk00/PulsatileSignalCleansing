import os
import subprocess
import sys

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

# Ensure numpy version is 1.19.x
try:
    import numpy as np
    numpy_version = np.__version__
    if numpy_version.split('.')[:2] != ['1', '19']:
        print(f"Current numpy version is {numpy_version}. Downgrading to 1.19.5...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.19.5"])
        # Reload numpy to ensure the downgraded version is used
        import importlib
        importlib.reload(np)
except ImportError:
    print("numpy is not installed. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.19.5"])
    import numpy as np

# Ensure pandas version is 1.4 or lower
try:
    import pandas as pd
    pandas_version = pd.__version__
    if float('.'.join(pandas_version.split('.')[:2])) > 1.4:
        print(f"Current pandas version is {pandas_version}. Downgrading to 1.4.0...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas==1.4.0"])
        # Reload pandas to ensure the downgraded version is used
        import importlib
        importlib.reload(pd)
except ImportError:
    print("pandas is not installed. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas==1.4.0"])
    import pandas as pd

def main():
    # Create an empty DataFrame with specified columns
    df = pd.DataFrame(columns=['patient', 'record', 'sig_len', 'time(s)', 'fs'])

    # Generate patient list (plist)
    raw_path = './Data_0Raw/MIMIC3/physionet.org/files/mimic3wdb/1.0'
    plist = [p for p in os.listdir(raw_path) if p[0] == '3']

    # Iterate through each patient folder
    for p in tqdm(plist, desc="Processing patients", unit="patient"):
        # Generate a list of patient folders
        patlist = [pat for pat in os.listdir(f'{raw_path}/{p}/') if pat[0] == '3']
        
        for pat in tqdm(patlist, desc=f"Processing patient {p}", unit="folder", leave=False):
            # Generate a list of records
            reclist = [rec[:-4] for rec in os.listdir(f'{raw_path}/{p}/{pat}/') if rec[-4:] == '.dat' and rec[7] == '_']

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
                    wfdata = wfdb.rdrecord(f'{raw_path}/{p}/{pat}/{rec}', channel_names=['ART', 'PLETH', 'II'])

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
                os.makedirs('./Data_0Raw/MIMIC3/raw_signals', exist_ok=True)

                # Save the dictionary as a compressed .npz file
                np.savez_compressed(f'./Data_0Raw/MIMIC3/raw_signals/{pat}.npz', **wfdb_dic)

    # Save the DataFrame to a CSV file
    df.to_csv('records_list_partial.csv', index=False)

if __name__ == "__main__":
    main()
