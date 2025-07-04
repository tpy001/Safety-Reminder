"""
    Sample benign data from VLSafe dataset
"""


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from datasets import load_dataset, Dataset
import random
import os
from src.utils import debug

# debug()

dataset_folder = "data/VLSafe/LVLM_NLF"
output_folder = "data/VLSafe_normal"
nums = 1000


# 加载数据
dataset = load_dataset(dataset_folder, data_files="helpfulness_nlf.jsonl", split="train")


sampled_dataset = dataset.shuffle(seed=42).select(range(nums))

# 处理数据为

1 + 1
# 保存为 jsonl 格式
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "helpful_sampled.jsonl")
sampled_dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)

print(f"Saved {nums} helpful samples to {output_file}")