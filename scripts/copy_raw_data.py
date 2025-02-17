import os
import shutil
import glob

def copy_csv_files(source_dir, dest_dir):
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    # Create a pattern to match all CSV files in the source directory
    pattern = os.path.join(source_dir, "*.csv")
    
    # Loop over each CSV file and copy it to the destination directory
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy(filepath, dest_path)
        print(f"Copied {filename} to {dest_dir}")

if __name__ == "__main__":
    source_directory = "/home/codespace/.cache/kagglehub/datasets/szrlee/stock-time-series-20050101-to-20171231/versions/3"
    destination_directory = "/workspaces/StockTradingAI-Project/raw"
    copy_csv_files(source_directory, destination_directory)
