"""
Sample data for VLSafe dataset and save the sampled data
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.dataset import VLSafe
from src.utils import set_seed
import pandas as pd
from PIL import Image
from src.utils import debug
import json

# debug()
set_seed(0)
num_samples = 300
output_dir = "data/VLSafe_sampled/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_dir = os.path.join(output_dir,"images")
question_dir = os.path.join(output_dir,"VLSafe")

if not os.path.exists(image_dir):
    os.makedirs(image_dir)

if not os.path.exists(question_dir):
    os.makedirs(question_dir)


dataset_config = {
    "_target_": "src.dataset.VLSafe",
    "data_path": "data/VLSafe/VLSafe",
    "data_files": "harmlessness_examine.jsonl",  # test dataset
    "split": "train",
    "question": "query",
    "chosen": "reference",
    "image": "image_id",
    "COCO_img_path": "data/VLSafe/images",
    "safe": False,
    "sample": num_samples,
    "seed": 0,
}


VLSafe_dataset = VLSafe(**dataset_config)


data = {
    "query": VLSafe_dataset["question"],
    "caption": VLSafe_dataset["caption"],
    "image_id": VLSafe_dataset["image"],
}

output_question_path = os.path.join(output_dir, "VLSafe","harmlessness_examine.jsonl")

output_image_path = os.path.join(output_dir,"images")


# save the question
with open(output_question_path, "w", encoding="utf-8") as f:
    for i in range(num_samples):
        image_path = VLSafe_dataset["image"][i]
        image_name = os.path.basename(image_path)
        new_image_path = os.path.join(output_image_path, image_name)
        image = Image.open(image_path)

        # save the sampled images
        image.save(new_image_path)

        # save the sampled questions
        item = {
            "image_id": image_name,
            "caption": data["caption"][i],
            "query": data["query"][i],
        }
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


     

