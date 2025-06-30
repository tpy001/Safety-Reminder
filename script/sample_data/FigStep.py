"""
Sample data for FigStep dataset and save the sampled data
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.dataset import FigStep
from src.utils import set_seed
import pandas as pd
from PIL import Image
from src.utils import debug

set_seed(0)
samples_per_category = 30
output_dir = "data/FigStep_sampled/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(os.path.join(output_dir,"question")):
    os.makedirs(os.path.join(output_dir,"question"))

if not os.path.exists(os.path.join(output_dir,"images")):
    os.makedirs(os.path.join(output_dir,"images"))

if not os.path.exists(os.path.join(output_dir,"images","SafeBench")):
    os.makedirs(os.path.join(output_dir,"images","SafeBench"))

dataset_config = {
    "_target_": "src.dataset.FigStep",
    "data_path": "data/FigStep/",
    "data_files": "question/safebench.csv",
    "split": "train",
    "question": "question",
    "ori_question": "instruction",
    "category": "category_name",
    "safe": False,
    "scenario": [
        "Illegal Activity",
        "Hate Speech",
        "Malware Generation",
        "Physical Harm",
        "Fraud",
        "Adult Content",
        "Privacy Violation",
        # "Legal Opinion",
        # "Financial Advice",
        # "Health Consultation",
    ],
    "prompt_type": "harmless",  # or "normal"
    "samples_per_category": samples_per_category
}


FigStep_dataset = FigStep(**dataset_config)


num_samples = len(FigStep_dataset["question"])
data = {
    "dataset": FigStep_dataset['dataset'],
    "category_id": FigStep_dataset['category_id'],  # 你可以改为实际 ID 映射
    "task_id": FigStep_dataset['task_id'],
    "category_name": FigStep_dataset["category"],
    "question": FigStep_dataset["ori_question"],
    "instruction": [""] * len(FigStep_dataset["ori_question"])
}

output_csv_path = os.path.join(output_dir, "question","safebench.csv")
output_image_path = os.path.join(output_dir,"images", "SafeBench")

# save the sampled text
df = pd.DataFrame(data)
df.to_csv(output_csv_path, index=False)
print(f"Saved CSV to {output_csv_path}")




# save the sampled image
for i in range(len(FigStep_dataset)):
    image_path = FigStep_dataset["image"][i]
    image_name = os.path.basename(image_path)
    image = Image.open(image_path)
    image.save(os.path.join(output_image_path, image_name))

print(f"Saved {len(FigStep_dataset)} images to {output_image_path}")