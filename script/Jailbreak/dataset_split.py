"""
Split the AdvBench dataset into train/val/test sets.
Total: 520 samples
Train dataset: 320 samples
Val dataset: 100 samples
Test dataset: 100 samples
"""

from datasets import load_dataset
import os

# --- 1. Load the original dataset ---
data_file_folder = "data/AdvBench"


# Load the CSV file
full_dataset = load_dataset("csv", data_files=f"{data_file_folder}/harmful_behaviors.csv")['train']

print(f"Total size of the original dataset: {len(full_dataset)}")

# --- 2. First Split: Separate the training set from the remainder ---
# We need 320 training samples, so the size of the remainder is 520 - 320 = 200.
# We can directly set test_size to the integer 200.
train_and_temp_split = full_dataset.train_test_split(test_size=200, seed=42)

# Assign the training set (320 samples)
train_dataset = train_and_temp_split['train']
# This is the remainder (200 samples), which we will use to create the validation and test sets
temp_dataset = train_and_temp_split['test']


# --- 3. Second Split: Split the remainder in half for validation and test sets ---
# We need to split the 200 samples into 100/100, so test_size is 0.5.
val_and_test_split = temp_dataset.train_test_split(test_size=0.5, seed=42)

# Assign the validation set (100 samples)
val_dataset = val_and_test_split['train']
# Assign the test set (100 samples)
test_dataset = val_and_test_split['test']


# --- 4. Verify the results and save to disk ---
print("\nDataset sizes after splitting:")
print(f"Train dataset: {len(train_dataset)}")
print(f"Validation dataset: {len(val_dataset)}")
print(f"Test dataset: {len(test_dataset)}")

# Check if the total count still matches
assert len(train_dataset) + len(val_dataset) + len(test_dataset) == len(full_dataset)

# Save the split datasets to disk
train_dataset.to_csv(f"{data_file_folder}/train.csv")
val_dataset.to_csv(f"{data_file_folder}/val.csv")
test_dataset.to_csv(f"{data_file_folder}/test.csv")

print("\nDatasets have been successfully split and saved to disk.")