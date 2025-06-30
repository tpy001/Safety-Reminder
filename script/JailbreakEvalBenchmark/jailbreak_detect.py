"""
Detect whether harmful prompts are generated using different judge models.
"""


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import json
from src.utils import debug,set_seed
import random
from src.utils import save_jailbreak_response
from datasets import load_dataset
from omegaconf import OmegaConf
from hydra.utils import instantiate

def load_judge_model(self,config_path, name = "evaluator"):
    config = OmegaConf.load(config_path)
    model_cfg = config.get(name,None)
    model = instantiate(model_cfg)
    return model


seed = 0
judge_model_name = "LLamaGuard3"
# judge_model_name = "MDJudge"

params = {
    "seed": seed,
    "benchmark_file": "JailbreakEvalBenchmark.json",
    "config_path":  f"configs/evaluator/{judge_model_name}.yaml",
    "output_file_name": os.path.join(os.path.dirname(__file__),f"detect_result_model={judge_model_name}_seed={seed}.json")
}

# Set the random seed
set_seed(params["seed"])

debug()


# Load the benchmark dataset
data = load_dataset("json", data_files=params["benchmark_file"], split='train')
data = data.rename_column("respnse", "respnses")

# Load the judge model
judge_model = load_judge_model(config_path=params["config_path"])

# Predict the safety of the responses
safe_pred = judge_model.judge(data["respnses"],data["question"])

# Save the predictions to a file
save_jailbreak_response(data["respnses"], data, safe_pred,output_file_path=params["output_file_name"])


