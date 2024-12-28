from datasets import load_dataset

# Define the dataset identifier
dataset_identifier = "Jagadeesh9580/semi-Voxpopuli"

# Load the dataset
# If it's private, add `use_auth_token=True`
dataset = load_dataset(dataset_identifier)

# Display the dataset structure
print(dataset)

# Save each split to a local file
for split in dataset.keys():  # Loop through splits (e.g., train, test, validation)
    print(f"Saving {split} split...")
    dataset[split].to_csv(f"semi_voxpopuli_{split}.csv")
    print(f"{split} split saved as semi_voxpopuli_{split}.csv")
