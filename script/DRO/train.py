import sys
import os
import argparse
from pathlib import Path
import hydra

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils import debug
from src.runner.DRO_runner import DRO_Runner
from src.utils import set_seed
from hydra.utils import instantiate

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@hydra.main(version_base=None,config_path="../../configs/", config_name="train_DRO.yaml")  
def main(cfg) -> None:  
    is_debug = cfg.get("debug", False)
    if is_debug:
        debug()

    # 1. Set seed
    seed = cfg.get("seed", 0)
    set_seed(seed)

    # 2. Load model
    model = instantiate(cfg.model)

    # 3. Load dataset
    train_dataset = instantiate(cfg.train_dataset)
    val_harmful_dataset = instantiate(cfg.val_harmful_dataset)
    val_normal_dataset = instantiate(cfg.val_normal_dataset)

    # 4. Load evaluator
    jailbreak_evaluator = instantiate(cfg.jailbreak_evaluator)
    normal_evaluator = instantiate(cfg.normal_evaluator)  

    # Initialize the Runner and start training
    runner = DRO_Runner(
        model = model,
        training_set= train_dataset,
        val_harmful_dataset= val_harmful_dataset,   
        val_normal_dataset = val_normal_dataset,
        jailbreak_evaluator= jailbreak_evaluator,
        normal_evaluator= normal_evaluator,
        train_config= cfg.train_config
    )
    runner.train()

if __name__ == "__main__":
    main()