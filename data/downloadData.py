import os
import shutil
import subprocess

# Define the repository URL and target folders
repo_url = "https://huggingface.co/datasets/Jagadeesh9580/semi-Voxpopuli"
output_folders = {
    "train": ".train",
    "test": ".test",
    "validation": ".validate"
}

def clone_repo(repo_url):
    # Clone the repository
    repo_name = repo_url.split("/")[-1]
    if os.path.exists(repo_name):
        print(f"Repository '{repo_name}' already exists. Removing it first...")
        shutil.rmtree(repo_name)

    print("Cloning the repository...")
    subprocess.run(["git", "clone", repo_url], check=True)
    return repo_name

def organize_dataset(repo_name):
    try:
        # Navigate to the cloned repository folder
        repo_path = os.path.join(os.getcwd(), repo_name)
        
        # Check if the repository folder exists
        if not os.path.exists(repo_path):
            raise FileNotFoundError(f"Repository folder '{repo_name}' not found.")
        
        # Create folders for splits
        for folder in output_folders.values():
            os.makedirs(folder, exist_ok=True)

        # Move files to respective folders
        for split, folder in output_folders.items():
            split_file = os.path.join(repo_path, f"{split}.csv")
            if os.path.exists(split_file):
                shutil.move(split_file, os.path.join(folder, f"{split}.csv"))
                print(f"Moved {split}.csv to {folder}")
            else:
                print(f"{split}.csv not found in the repository.")

        # Clean up the cloned repository folder
        print("Removing cloned repository folder...")
        shutil.rmtree(repo_path)

    except Exception as e:
        print(f"An error occurred during dataset organization: {e}")

def cleanup_unrelated_files():
    # Remove unrelated files and folders in the current directory
    current_dir = os.getcwd()
    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)
        if item not in output_folders.values():
            print(f"Removing {item_path}...")
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

if __name__ == "__main__":
    try:
        # Clone the repository
        repo_name = clone_repo(repo_url)

        # Organize the dataset
        organize_dataset(repo_name)

        # Clean up any remaining unrelated files
        cleanup_unrelated_files()

        print("Dataset processing complete.")
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
