
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import set_seed
from omegaconf import OmegaConf
from hydra.utils import instantiate
import json
from src.utils import save_jailbreak_response,debug



 
if __name__ == "__main__":
    seed = 0
    set_seed(seed)

    # config_path = "configs/evaluator/LLamaGuard3.yaml"
    config_path = "configs/evaluator/MDJudge.yaml"
    input_file = "./response.json"
    output_file = "./results.json"
    name = "evaluator"

    # Load the config file
    config = OmegaConf.load(config_path)
    model_cfg = config.get(name,None)
    model = instantiate(model_cfg)

    # load the input dsata from the file
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)


    responses = [item["response"] for item in data]
    questions = [item["question"] for item in data]
    answers = [item["answer"] for item in data]
    category = [item["category"] for item in data]

    data = {
        "responses": responses,
        "question": questions,
        "answer": answers,
        "category": category
    }

    # load the test data from the files
    safe_pred = model.judge(responses,questions)
    
  
    # save the results to the file
    save_jailbreak_response(responses, data, safe_pred, output_file_path=output_file)

