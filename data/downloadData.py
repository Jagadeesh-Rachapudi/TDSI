import os
import subprocess
from zipfile import ZipFile
import random
import shutil
import stat


# Constants
REPO_URL = "https://huggingface.co/datasets/Jagadeesh9580/semi-Voxpopuli"
REPO_DIR = "data/semi-Voxpopuli"
TOY_DATA_DIR = os.path.join(REPO_DIR, "toyData")
TRAIN_DIR = "data/"
TEST_DIR = "data/"
VALIDATE_DIR = "data/"

# Specify lengths (set to None to move all files)
TRAIN_LEN = 500  # Set to an integer to move a specific number of training files
TEST_LEN = 50   # Set to an integer to move a specific number of test files
VALIDATE_LEN = 50  # Set to an integer to move a specific number of validation files

def clone_repo():
    # Clone the repository
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)


def remove_git_files():
    # Use shutil to remove the .git folder with forced permissions
    git_dir = os.path.join(REPO_DIR, ".git")
    if os.path.exists(git_dir):
        def onerror(func, path, exc_info):
            # Change file permissions and retry
            os.chmod(path, stat.S_IWRITE)
            func(path)

        shutil.rmtree(git_dir, onerror=onerror)

def extract_and_move_files(zip_path, target_dir, move_len=None):
    # Extract and move specified number of files
    temp_extract_dir = "./temp_extracted"
    os.makedirs(temp_extract_dir, exist_ok=True)
    
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)
    
    all_files = os.listdir(temp_extract_dir)
    if move_len is not None:
        all_files = random.sample(all_files, min(len(all_files), move_len))
    
    os.makedirs(target_dir, exist_ok=True)
    for file_name in all_files:
        src = os.path.join(temp_extract_dir, file_name)
        dest = os.path.join(target_dir, file_name)
        shutil.move(src, dest)
    
    # Clean up temporary directory
    shutil.rmtree(temp_extract_dir)

def process_toy_data():
    # Process toy data files and move them to respective directories
    toy_data_files = {
        "train.zip": (TRAIN_DIR, TRAIN_LEN),
        "test.zip": (TEST_DIR, TEST_LEN),
        "validate.zip": (VALIDATE_DIR, VALIDATE_LEN),
    }

    for file_name, (target_dir, move_len) in toy_data_files.items():
        zip_path = os.path.join(TOY_DATA_DIR, file_name)
        if os.path.exists(zip_path):
            extract_and_move_files(zip_path, target_dir, move_len)

def cleanup_repo():
    # Remove all other files from the semi-Voxpopuli repo except ./train, ./test, ./validate
    for item in os.listdir(REPO_DIR):
        item_path = os.path.join(REPO_DIR, item)
        if item not in ["train", "test", "validate"]:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
                shutil.rmtree(".\semi-Voxpopuli", ignore_errors=True)
            else:
                os.remove(item_path)
    # Path to the folder
    folder_path = "./data/semi-Voxpopuli"

    # Check if the folder exists
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  
        print(f"Deleted folder: {folder_path}")
    else:
        print(f"Folder does not exist: {folder_path}")


def rename():
    base_path = "./data"  # Change this to your base directory if different
    folders_to_rename = {
        "train_ultimate": "train",
        "test_ultimate": "test",
        "validate_ultimate": "validate",
    }

    # Rename the folders
    for old_name, new_name in folders_to_rename.items():
        old_path = os.path.join(base_path, old_name)
        new_path = os.path.join(base_path, new_name)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"Renamed {old_path} to {new_path}")
        else:
            print(f"Folder does not exist: {old_path}")


if __name__ == "__main__":
    # Step 1: Clone the repo
    clone_repo()
    
    # Step 2: Remove all .git files
    remove_git_files()
    
    # Step 3: Extract and move files to respective directories
    process_toy_data()
    
    # Step 4: Move the extracted data folders to the current working directory
    for dir_name in ["train", "test", "validate"]:
        src = os.path.join(REPO_DIR, dir_name)
        dest = f"./{dir_name}"
        if os.path.exists(src):
            shutil.move(src, dest)
    
    # Step 5: Cleanup the repo
    cleanup_repo()
    rename()