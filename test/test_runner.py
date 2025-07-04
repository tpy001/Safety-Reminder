"""
Test the jailbreak success rate on FigStep dataset."
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

class TestRunner:
    config_dir = "../../configs/"
    config_name = "llava_PT.yaml"
    config = None
    seed = 0
    runner = None


    @classmethod
    def setup_class(cls):
        set_seed(cls.seed)
        cls.runner = Runner(cls.config_dir,cls.config_name)
       

    @classmethod
    def teardown_class(cls):
        del cls.runner
        torch.cuda.empty_cache()  

    def test_is_not_none(self):
        assert self.runner is not None, "Runner is None."

    def test_train(self):
        self.runner.train()