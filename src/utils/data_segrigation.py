import os
import random
import shutil
from pathlib import Path

def split_dataset(data_folder, output_folder, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Ensure the ratios add up to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must add up to 1"

    # Create output directories
    train_path = os.path.join(output_folder, "train")
    val_path = os.path.join(output_folder, "validate")
    test_path = os.path.join(output_folder, "test")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Get all files from the data folder
    data_folder = Path(data_folder)
    all_files = list(data_folder.glob("**/*"))  # Includes files in all subdirectories
    all_files = [f for f in all_files if f.is_file()]

    # Shuffle the files
    random.shuffle(all_files)

    # Split the data
    total_files = len(all_files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)

    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    def copy_files(files, destination):
        for file in files:
            relative_path = file.relative_to(data_folder)
            dest_file_path = Path(destination) / relative_path
            os.makedirs(dest_file_path.parent, exist_ok=True)
            shutil.copy2(file, dest_file_path)

    # Copy files to respective directories
    copy_files(train_files, train_path)
    copy_files(val_files, val_path)
    copy_files(test_files, test_path)

    print(f"Data split completed!")
    print(f"Train files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")

# Example usage
if __name__ == "__main__":
    data_folder = "path/to/your/data/folder"  # Replace with your data folder path
    output_folder = "path/to/output/folder"  # Replace with your output folder path
    split_dataset(data_folder, output_folder)
