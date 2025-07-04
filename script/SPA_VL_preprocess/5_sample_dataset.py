"""
Sample data from the processed dataset and save the sampled data to a new file.

"""

import os
import sys
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from datasets import load_dataset, Dataset, concatenate_datasets
import multiprocessing
from tqdm import tqdm
from src.utils import debug

# debug()

# It's good practice to get the number of CPUs available
# Capping at 8 as per your original code.
num_cpus = min(multiprocessing.cpu_count(), 8)
print(f"Utilizing {num_cpus} CPU cores for processing.")

# --- Step 1: Define Categories and Sample Numbers (Unchanged) ---
categories = {
    "Illegal Activity": [
        "Malicious Use/Illegal activities/drug-related crimes",
        "Malicious Use/Illegal activities/environmental crimes",
        "Malicious Use/Illegal activities/financial crimes",
        "Malicious Use/Illegal activities/theft",
        "Malicious Use/Illegal activities/traffic and driving offenses",
        "Malicious Use/Security threats/biological and chemical harms",
        "Malicious Use/Security threats/weapon",
    ],
    "Hate Speech": [
        "Representation & ToxicityHarms/Toxic/bully",
        "Representation & ToxicityHarms/Toxic/harass",
        "Representation & ToxicityHarms/Toxic/hate",
        "Representation & ToxicityHarms/Toxic/insult",
        "Representation & ToxicityHarms/Toxic/terrorism",
        "Representation & ToxicityHarms/Toxic/threaten",
        "Representation & ToxicityHarms/Toxic/violent",
        "Representation & ToxicityHarms/Unfair/ageism",
        "Representation & ToxicityHarms/Unfair/disablity discrimination",
        "Representation & ToxicityHarms/Unfair/geographical discrimination",
        "Representation & ToxicityHarms/Unfair/racial",
        "Representation & ToxicityHarms/Unfair/religion",
        "Representation & ToxicityHarms/Unfair/sexism",
    ],
    "Malware Generation":[
        "Malicious Use/Security threats/malware",
        "Malicious Use/Security threats/cyber",
    ],
    "physical harm": [
        "Human Autonomy & IntegrityHarms/Persuasion and manipulation/physical harm",
    ],
    "Economic Harm": [
        "Socioeconomic Harms/Risky financial practices/gamble",
        "Socioeconomic Harms/Risky financial practices/multi-level marketing",
        "Socioeconomic Harms/Risky financial practices/paypal lending",
    ],
    "Fraud": [
        "Malicious Use/Fraud or deceptive action/academic dishonesty",
        "Malicious Use/Fraud or deceptive action/disinformation",
        "Malicious Use/Fraud or deceptive action/fake news",
        "Malicious Use/Fraud or deceptive action/financial fraud",
        "Malicious Use/Fraud or deceptive action/pseudo-pharmaceuticals",
        "Malicious Use/Fraud or deceptive action/rumors",
        "Malicious Use/Fraud or deceptive action/scamns",
        "Malicious Use/Fraud or deceptive action/spam information",
    ],
    "Privacy Violation": [
        "Information & Safety Harms/Privacy/personal privacy"
    ],
    "Adult Content": []
}

# sample total 1000 data
scale = 1
categories_nums = {
    "Illegal Activity": 200 * scale,
    "Hate Speech": 150 * scale,
    "Malware Generation": 150 * scale,
    "physical harm": 100 * scale,
    "Economic Harm": 100 * scale,
    "Fraud": 200 * scale,
    "Privacy Violation": 100 * scale,
    "Adult Content": 0
}


data_path = "./SPA_VL_converted_final_11759.json"
output_data_file = "./SPA_VL_sampled.json"

# --- Main Logic ---

# Step 1: Load the full dataset from the specified JSON file.
print(f"Loading dataset from {data_path}...")
try:
    # Assumes the data is in the 'train' split, which is default for single files.
    full_dataset = load_dataset('json', data_files=data_path, split='train')
    print(f"Dataset loaded successfully. Total original samples: {len(full_dataset)}")
except Exception as e:
    print(f"Error loading dataset: {e}. Please check the file path and format.")
    sys.exit(1) # Exit if the dataset cannot be loaded.

# A list to hold the smaller, sampled datasets from each category.
sampled_datasets = []

# Step 2: Iterate through each category to filter and sample the data.
for category, subcategories in tqdm(categories.items(), desc="Sampling across categories"):
    n_samples_to_get = categories_nums.get(category, 0)

    # Skip if no samples are requested or no subcategories are defined.
    if n_samples_to_get == 0 or not subcategories:
        print(f"\nSkipping '{category}' (Samples requested: {n_samples_to_get}, Subcategories defined: {bool(subcategories)})")
        continue

    # Use a set for faster membership checking.
    subcategories_set = set(subcategories)
    
    # Filter the dataset to get only the items matching the current category's taxonomies.
    # This is done in parallel using the available CPU cores.
    triplets_for_main_category = {tuple(path.split('/')) for path in subcategories_set if len(path.split('/')) == 3}
    
    filtered_dataset = full_dataset.filter(
        lambda x: (x["class1"], x["class2"], x["class3"]) in triplets_for_main_category and x["safe"] == 0,
    )

    num_available = len(filtered_dataset)
    print(f"\nProcessing '{category}': Found {num_available} available samples.")

    # If no samples are found for a category, issue a warning and continue.
    if num_available == 0:
        print(f"Warning: No samples found for category '{category}'.")
        continue
    
    # If the available samples are fewer than requested, take all available samples.
    if num_available < n_samples_to_get:
        print(f"Warning: For '{category}', requested {n_samples_to_get} but only {num_available} are available. Taking all.")
        n_samples_to_get = num_available

    # Shuffle the filtered data and select the desired number of samples.
    # A seed is used to make the random shuffling reproducible.
    sampled_subset = filtered_dataset.shuffle(seed=42).select(range(n_samples_to_get))
    sampled_datasets.append(sampled_subset)
    print(f"Successfully sampled {len(sampled_subset)} items for '{category}'.")

# Step 3: Combine all the sampled subsets into a single dataset.
if sampled_datasets:
    print("\nConcatenating all sampled datasets...")
    final_dataset = concatenate_datasets(sampled_datasets)
    
    final_dataset = final_dataset.remove_columns(['safe', 'chosen_safe'])
    # Step 4: Save the final combined dataset to a new JSON file.
    print(f"Saving final sampled dataset with {len(final_dataset)} records to {output_data_file}...")
    final_dataset.to_json(output_data_file, orient='records', lines=True, force_ascii=False)
    print("Sampling process completed successfully!")
else:
    print("\nNo data was sampled. The output file was not created.")


    

        


