import subprocess

def download_data():
    url = "https://physionet.org/files/mimic3wdb/1.0/"
    wget_command = ["wget", "-r", "-N", "-c", "-np", "--no-check-certificate", url]
    
    try:
        result = subprocess.run(wget_command, check=True)
        print("Download completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_data()
