"""
Sample data for MMSafetyBench dataset and save the sampled data
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.dataset import MMSafetyBench
from src.utils import set_seed
import pandas as pd
from PIL import Image
from src.utils import debug
from copy import deepcopy
import shutil
import json

set_seed(0)
samples_per_category = 50
output_dir = "data/MMSafetyBench_sampled/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(os.path.join(output_dir,"processed_questions")):
    os.makedirs(os.path.join(output_dir,"processed_questions"))

img_path = os.path.join(output_dir,"imgs")

if not os.path.exists(img_path):
    os.makedirs(img_path)


dataset_config_TYPO = {
    "_target_": "src.dataset.MMSafetyBench",
    "data_path": "data/MMSafetyBench/",
    "split": "test",
    "safe": False,
    "image_type": "TYPO",
    "scenario": [
        "01-Illegal_Activitiy",
        "02-HateSpeech",
        "03-Malware_Generation",
        "04-Physical_Harm",
        "05-EconomicHarm",
        "06-Fraud",
        "07-Sex",
        "08-Political_Lobbying",
        "09-Privacy_Violence",
        "10-Legal_Opinion",
        "11-Financial_Advice",
        "12-Health_Consultation",
        "13-Gov_Decision"
    ],
    "seed": 0,
    "samples_per_category": samples_per_category
}

dataset_config_SD = deepcopy(dataset_config_TYPO)
dataset_config_SD["image_type"] = "SD"

dataset_config_SD_TYPO = deepcopy(dataset_config_TYPO)
dataset_config_SD_TYPO["image_type"] = "SD+TYPO"

MMSafetyBench_TYPO = MMSafetyBench(**dataset_config_TYPO)
MMSafetyBench_SD = MMSafetyBench(**dataset_config_SD)
MMSafetyBench_SD_TYPO = MMSafetyBench(**dataset_config_SD_TYPO)


num_samples = len(MMSafetyBench_TYPO["question"])
data = {
    "Question": MMSafetyBench_TYPO['ori_question'],  # 你可以改为实际 ID 映射
    "Rephrased Question(SD)": MMSafetyBench_SD['question'],
    "Rephrased Question": MMSafetyBench_SD_TYPO['question'],
    "category": MMSafetyBench_TYPO["category"],
    "TYPO_img": MMSafetyBench_TYPO['image'],
    "SD_img": MMSafetyBench_SD['image'],
    "SD+TYPO_img": MMSafetyBench_SD_TYPO['image'],
    "id": MMSafetyBench_TYPO['id']
}

output_csv_path = os.path.join(output_dir, "question","safebench.csv")
output_image_path = os.path.join(output_dir,"images", "SafeBench")

category_list = list(set(MMSafetyBench_TYPO["category"]))
# save the sampled image
for category in category_list:
    if not os.path.exists(os.path.join(img_path,category)):
        os.makedirs(os.path.join(img_path,category))
        dst_dir_SD = os.path.join(img_path,category,"SD") 
        dst_dir_TYPO = os.path.join(img_path,category,"TYPO")
        dst_dir_SD_TYPO = os.path.join(img_path,category,"SD_TYPO")
        os.makedirs(dst_dir_SD)
        os.makedirs(dst_dir_TYPO)
        os.makedirs(dst_dir_SD_TYPO)
        
        for i in range(len(MMSafetyBench_TYPO["question"])): 
            if MMSafetyBench_TYPO["category"][i] == category:
                TYPO_img_path = MMSafetyBench_TYPO["image"][i]
                SD_img_path = MMSafetyBench_SD["image"][i]
                SD_TYPO_img_path = MMSafetyBench_SD_TYPO["image"][i]
                shutil.copy(TYPO_img_path, dst_dir_TYPO)  # copy the image to the target directory
                shutil.copy(SD_img_path, dst_dir_SD)  # copy the image to the target directory
                shutil.copy(SD_TYPO_img_path, dst_dir_SD_TYPO)  # copy the image to the target directory

# save the questions
question_path = os.path.join(output_dir,"processed_questions")
for category in category_list:
    _data = {}
    for i in range(len(MMSafetyBench_TYPO["question"])):
        if data["category"][i] == category:
            id = data["id"][i]
            _data[id] = {
                "Question": data["Question"][i],
                "Rephrased Question(SD)": data["Rephrased Question(SD)"][i],
                "Rephrased Question": data["Rephrased Question"][i],
                "category": data["category"][i]
            } 

    # save the json file
    output_file = os.path.join(question_path, f"{category}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(_data, f, ensure_ascii=False, indent=4)
