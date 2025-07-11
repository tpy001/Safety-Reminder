import sys
import os
import argparse
from pathlib import Path
import hydra

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils import debug
from src.runner.SAPT_runner import SAPT_Runner
from src.utils import set_seed
from hydra.utils import instantiate

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@hydra.main(version_base=None,config_path="../../configs/", config_name="train.yaml")  
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
    val_dataset = instantiate(cfg.val_dataset)

    # 4. Load evaluator
    evaluator = instantiate(cfg.evaluator)

    # Initialize the Runner and start training
    runner = SAPT_Runner(
        model = model,
        training_set= train_dataset,
        val_dataset= val_dataset,   
        evaluator= evaluator,
        train_config= cfg.train_config
    )
    runner.train()

if __name__ == "__main__":
    main()