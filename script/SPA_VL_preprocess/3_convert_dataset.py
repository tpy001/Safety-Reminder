"""
preprocess data from the specified categories of interest.

Step 1: Define the categories of interest and the desired number of samples per category.  
Step 2: Filter the dataset to include only data belonging to the specified categories.  
Step 3: Use the Llama-Guard Vision model to filter out safe data (we filter out the data wherethe rejected answer is safe)..  
Step 4: Save the preprocessed data to the designated data folder.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from datasets import load_dataset, Dataset, concatenate_datasets
import multiprocessing
from src.evaluator import RuleBasedEvaluator, LLamaGuard3, MDJudge,LlamaGuard3Vision
from tqdm import tqdm
from datasets import concatenate_datasets

# It's good practice to get the number of CPUs available
num_cpus = min( multiprocessing.cpu_count(),8)
print(f"Utilizing {num_cpus} CPU cores for processing.")


from src.utils import debug

# debug()

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
scale =2 
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


data_path = "data/SPA_VL/"
output_data_file = "./SPA_VL_converted_final.json"

# --- Main Logic ---

# If the final file exists, load it directly.
if os.path.exists(output_data_file):
    print(f"Loading existing data from {output_data_file}")
    converted_data = load_dataset("json", data_files={"train": output_data_file})["train"]

else:
    # Load the original dataset
    print("Loading original dataset...")
    dataset = load_dataset("parquet", data_files={"train": os.path.join(data_path, "train_converted", "data.parquet")})["train"]
    # dataset = dataset.remove_columns(["image"])

    # Prepare the set of all valid category triplets for efficient lookup
    triplets_to_include = set()
    for value in categories.values():
        for path in value:
            parts = path.split("/")
            if len(parts) == 3:
                triplets_to_include.add(tuple(parts))

    # --- Step 2 & 3: Optimized Filtering with Multiprocessing ---
    print("Filtering dataset in a single pass using multiple processes...")

    # This function checks if a row's category is in our target set.
    def is_category_of_interest(example):
        return (example["class1"], example["class2"], example["class3"]) in triplets_to_include

    # Apply the filter ONCE, using multiple processes.
    # This is much more efficient than filtering in a loop.
    converted_data = dataset.filter(
        is_category_of_interest,
        num_proc=num_cpus  # Use all available CPU cores
    )

    print(f"Found {len(converted_data)} total samples matching the specified categories.")

    if not converted_data:
        print("⚠️ No data matched the specified categories.")
        # Create an empty dataset if nothing is found
        converted_data = Dataset.from_dict({col: [] for col in dataset.column_names})

    # --- Step 5: Save the preprocessed data ---
    print(f"Saving filtered dataset to {output_data_file}")
    converted_data.to_json(output_data_file)

# The subsequent steps (Llama-Guard filtering, sampling, etc.) would follow from here.

# --- Step 3 & 4: Filter by Safety Model and Sample ---
print("\n--- Starting Safety Evaluation and Sampling ---")
batch_size = 16

# evaluator = LlamaGuard3Vision(
#     model_path="/data3/tangpeiyuan/LLM/weights/LlamaGuard3Vision/",
#     batchsize = batch_size
# )

evaluator = LLamaGuard3(
     model_path="/data3/tangpeiyuan/LLM/weights/Llama3-Guard-8b/",
     batchsize = batch_size
 )

if not "safe" in converted_data.column_names:
    converted_data = converted_data.add_column("safe", [-1] * len(converted_data))
if "id" not in converted_data.column_names:
    converted_data = converted_data.add_column("id", list(range(len(converted_data))))

# This list will hold the datasets for each category that meet the criteria

for main_category, sub_categories in categories.items():
    target_num = categories_nums.get(main_category, 0)
    
    final_datasets = Dataset.from_dict({col: [] for col in converted_data.column_names})
    print(f"\nProcessing Category: '{main_category}'. Target unsafe samples: {target_num}")

    if target_num == 0:
        print("Target is 0, skipping.")
        continue

    # Get all triplets for the current main category
    triplets_for_main_category = {tuple(path.split('/')) for path in sub_categories if len(path.split('/')) == 3}
    
    sub_datasets = converted_data.filter(
        lambda x: (x["class1"], x["class2"], x["class3"]) in triplets_for_main_category,
        num_proc=num_cpus
    )

    current_num = len(sub_datasets.filter(lambda x: x["safe"] == 0))
    print(f"Current unsafe samples: {current_num}")
    if current_num >= target_num:
        continue

   
    
    # Filter the dataset to get only the items for the current category
    category_subset = sub_datasets.filter(
        lambda x: x["safe"] == -1,
        num_proc=num_cpus
    )

    if len(category_subset) == 0:
        print("No data found for this category after filtering. Skipping.")
        continue
        
    print(f"Found {len(category_subset)} items. Evaluating with safety model...")

    # Use tqdm for a progress bar
    for i in range(0, len(category_subset), batch_size):
        batch_dict = category_subset[i:i+batch_size]

        # image_paths = [
        #     os.path.join(
        #         data_path, "train_converted", "images",
        #         batch_dict["class1"][j],
        #         batch_dict["class2"][j],
        #         batch_dict["class3"][j],
        #         batch_dict["image_name"][j]
        #     )
        #     for j in range(len(batch_dict["question"]))
        # ]

        _data = {
            "question": batch_dict["question"],
            # "image_path": image_paths,
        }

        # Run the evaluator
        safety_result = evaluator.judge(answers=batch_dict["rejected"], data=_data)

        # Add 'safe' label
        batch_dict["safe"] = [1 if x == "true" else 0 for x in safety_result]

        # Convert back to Dataset
        batch_with_label = Dataset.from_dict(batch_dict)

        # Append to final_datasets
        final_datasets = concatenate_datasets([final_datasets, batch_with_label])

        if i % 100 == 0:
            id_to_safe = dict(zip(final_datasets["id"], final_datasets["safe"]))
            converted_data = converted_data.map(
                lambda x: {"safe": id_to_safe.get(x["id"], x["safe"])}
            )

            if os.path.exists(output_data_file):
                os.remove(output_data_file)
            converted_data.to_json(
                output_data_file, 
            )
            
            final_datasets = Dataset.from_dict({col: [] for col in converted_data.column_names})

    #############################
    id_to_safe = dict(zip(final_datasets["id"], final_datasets["safe"]))
    converted_data = converted_data.map(
        lambda x: {"safe": id_to_safe.get(x["id"], x["safe"])}
    )

    if os.path.exists(output_data_file):
        os.remove(output_data_file)
    converted_data.to_json(
        output_data_file, 
    )
    #############################


final_datasets = Dataset.from_dict({col: [] for col in converted_data.column_names})

# Filter out safe data; only unsafe data will be retained.
unsafe_data = converted_data.filter(lambda x: x["safe"] == 0, num_proc=num_cpus)

unsafe_data.to_json(output_data_file.replace(".json", "_unsafe.json"))




        
      

            
   

