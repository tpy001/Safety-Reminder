"""
Test the jailbreak success rate on FigStep dataset."
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import hydra
from omegaconf import OmegaConf
from hydra import initialize, compose
from src.utils import debug
from pathlib import Path
from hydra.utils import instantiate
from script.generate import generate_text
from src.utils import set_seed
import torch
from src.runner.runner import Runner

# debug()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

config_dir = "../../configs/"
config_name = "llava_PT_training.yaml"
config = None
runner = None

runner = Runner(config_dir,config_name)
    

runner.train()