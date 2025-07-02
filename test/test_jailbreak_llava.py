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

class BaseTestJailbreakLlaVA:
    config_dir = "../configs/"
    config_name = ""
    config = None
    model = None
    dataset = None
    evaluator = None
    seed = 0
    expected_output =  None


    @classmethod
    def setup_class(cls):
        set_seed(cls.seed)
        with initialize(config_path=str(cls.config_dir), version_base=None):
            cfg = compose(config_name=cls.config_name)
            cls.model = instantiate(cfg.model.model)
            cls.dataset = instantiate(cfg.dataset.dataset) 
            cls.evaluator = instantiate(cfg.evaluator.evaluator) 

    @classmethod
    def teardown_class(cls):
        del cls.model
        del cls.evaluator
        torch.cuda.empty_cache()  

    def test_is_not_none(self):
        assert self.model is not None, "Model is None."
        assert self.dataset is not None, "Dataset is None."
        assert self.evaluator is not None, "Evaluator is None."

    def test_jailbreak_detection(self):
        test_data = self.dataset[:4]
        output_text = generate_text(
            self.model,
            test_data['question'],
            test_data['image'],
            use_image=True,
            batch_size=2
        )
        safe_pred = self.evaluator.judge(answers=output_text, data=test_data['ori_question'])

        expected_output = self.expected_output
        for i in range(len(safe_pred)):
            assert safe_pred[i].lower() == str(expected_output[i]).lower(), (
                f"Prediction is not correct. Expected {expected_output[i]}, "
                f"but got {safe_pred[i]} for prompt {test_data['question'][i]}"
            )


class TestLlavaFigStep(BaseTestJailbreakLlaVA):
    config_name = "llava_FigStep"
    expected_output =  ["false", "false", "false", "false"]

    

   
class TestLlavaMMSafetyBench(BaseTestJailbreakLlaVA):
    config_name = "llava_MMSafetyBench"
    expected_output =  ["false", "false", "false", "false"]



   

class TestLlavaVLSafe(BaseTestJailbreakLlaVA):
    config_name = "llava_VLSafe"
    expected_output =  ['false', 'true', 'false', 'false']



class TestSPA_VL(BaseTestJailbreakLlaVA):
    config_name = "llava_SPA_VL.yaml"
    expected_output =  ['true', 'true', 'false', 'true']

class TestPGDAttack(BaseTestJailbreakLlaVA):
    config_name = "llava_PGDAttackDataset.yaml"
    expected_output =  ['false', 'false', 'false', 'false']



   