"""
Sample 100 question-response pairs from each of the 5 datasets: FigStep, MMSafetyBench, VLSafe, SPA-VL, and AdvBench+VisualAdvAttack, for a total of 500 samples. 
This response is generated by LlaVA-1.5-7B model
"""


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import json
from src.utils import debug,set_seed
import random
from src.utils import save_jailbreak_response

seed = 0
set_seed(seed)

debug()

# Load the data from the given file
with open("script/JailbreakEvalBenchmark/FigStep.json",'r') as f:
    FigStep_data = json.load(f)

with open("script/JailbreakEvalBenchmark/MMSafetyBench.json",'r') as f:
    MMSafetyBench_data = json.load(f)

with open("script/JailbreakEvalBenchmark/VLSafe.json",'r') as f:
    VLSafe_data = json.load(f)

with open("script/JailbreakEvalBenchmark/SPA_VL.json",'r') as f:
    SPA_VL_data = json.load(f)

with open("script/JailbreakEvalBenchmark/AdvBench_with_adv_attack.json",'r') as f:
    AdvBench_data = json.load(f)

data = []

# Sample 100 question-response pairs from each of the 5 datasets
for item in [FigStep_data, MMSafetyBench_data, VLSafe_data, SPA_VL_data, AdvBench_data]:
    sampled_data = random.sample(item, 100)
    data.extend(sampled_data)


# Save the data to disk
output_file_name = "JailbreakEvalBenchmark.json"

response = [ data[i]['response'] for i in range(len(data))]
question = [ data[i]['question'] for i in range(len(data))]
answer = [ data[i]['answer'] for i in range(len(data))]
category = [ data[i]['category'] for i in range(len(data))]
image_path = [ data[i]['image_path'] for i in range(len(data))]

_data ={
    "question": question,
    "answer": answer,
    "category": category,
}

save_jailbreak_response(response, _data, image_path = image_path,output_file_path=output_file_name)

