import subprocess
import os

def download_data(download_dir):
    url = "https://physionet.org/files/mimic3wdb/1.0/"
    wget_command = ["wget", "-r", "-N", "-c", "-np", "--no-check-certificate", "-P", download_dir, url]
    
    try:
        result = subprocess.run(wget_command, check=True)
        print("Download completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_dir = "./Data_0Raw/MIMIC3/"  # Specify your download directory here
    os.makedirs(download_dir, exist_ok=True)  # Create the directory if it does not exist
    download_data(download_dir)
