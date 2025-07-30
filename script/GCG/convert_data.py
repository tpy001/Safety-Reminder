
from datasets import load_dataset
import json
import os


# data_folder = "work_dirs/GCG/llava"
data_folder = "work_dirs/GCG/DeepSeekVL"
output_path ="./GCG.csv"


data_paths = sorted(
    [os.path.join(data_folder, f) 
     for f in os.listdir(data_folder) if f.endswith(".json")]
)
adv_suffix_list = []
last_suffix_list = []
questions = []

for path in data_paths:
    data = json.load(open(path, "r"))
    adv_suffix = data["best_string"]
    last_suffix = data["strings"][-1]
    adv_suffix_list.append(adv_suffix)
    last_suffix_list.append(last_suffix)
    questions.append(data["question"])


dataset = load_dataset("csv", data_files="script/GCG/harmful_behaviors.csv")["train"]

adv_questions = [ questions[i] + " " + adv_suffix_list[i] for i in range(len(questions))]
adv_questions2 = [ questions[i] + " " + last_suffix_list[i] for i in range(len(questions))]

# import as csv file
import csv

with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["goal", "target", "adv_suffix"])  # 写入表头
    for goal, target, suffix in zip(adv_questions, dataset["target"], adv_suffix_list):
        writer.writerow([goal, target, suffix])
    
    for goal, target, suffix in zip(adv_questions2, dataset["target"], last_suffix_list):
        writer.writerow([goal, target, suffix])

