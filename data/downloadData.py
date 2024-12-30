import os
import shutil
import subprocess
import platform

# Define the repository URL and target folders
repo_url = "https://huggingface.co/datasets/Jagadeesh9580/semi-Voxpopuli"
output_folders = {
    "train": "./train",
    "test": "./test",
    "validate": "./validate"
}

def clone_repo(repo_url):
    """Clone the repository."""
    repo_name = repo_url.split("/")[-1]
    if os.path.exists(repo_name):
        print(f"Repository '{repo_name}' already exists. Removing it first...")
        shutil.rmtree(repo_name)

    print("Cloning the repository...")
    subprocess.run(["git", "clone", repo_url], check=True)
    return repo_name

def remove_git_folder(repo_name):
    """Remove the .git folder using subprocess."""
    git_folder_path = os.path.join(os.getcwd(), repo_name, ".git")
    if os.path.exists(git_folder_path):
        print("Removing .git folder...")
        try:
            if platform.system() == "Windows":
                subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", git_folder_path], check=True)
            else:
                subprocess.run(["rm", "-rf", git_folder_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to remove .git folder: {e}")
            raise

def organize_dataset(repo_name):
    """Organize dataset files into respective folders."""
    try:
        # Define source paths
        repo_path = os.path.join(os.getcwd(), repo_name, "data")
        if not os.path.exists(repo_path):
            raise FileNotFoundError(f"Repository data folder '{repo_path}' not found.")

        # Create output folders and move data
        for split, target_folder in output_folders.items():
            source_path = os.path.join(repo_path, split)
            if os.path.exists(source_path):
                os.makedirs(target_folder, exist_ok=True)
                print(f"Moving {split} data to {target_folder}...")
                for file in os.listdir(source_path):
                    shutil.move(os.path.join(source_path, file), target_folder)
            else:
                print(f"No data found for '{split}' in the repository.")

        # Remove the .git folder using subprocess
        remove_git_folder(repo_name)

        # Remove the cloned repository folder
        print(f"Removing the cloned repository folder: {os.path.join(os.getcwd(), repo_name)}")
        shutil.rmtree(os.path.join(os.getcwd(), repo_name))

    except Exception as e:
        print(f"An error occurred during dataset organization: {e}")

def main():
    try:
        # Clone the repository
        repo_name = clone_repo(repo_url)

        # Organize the dataset
        organize_dataset(repo_name)

        print("Dataset processing complete.")
    except subprocess.CalledProcessError as e:
        print(f"Subprocess error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
