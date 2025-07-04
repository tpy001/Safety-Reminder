"""
Test the correctness of the dataset class.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import debug
from omegaconf import OmegaConf
from src.dataset.FigStep import FigStep
from hydra.utils import instantiate
from src.utils import set_seed
from copy import deepcopy

debug()
class BaseTestDataset:
    config_path = "configs/dataset/default.yaml"
    config = None
    data_list = None
    name = "dataset"
    
    @classmethod
    def setup_class(cls):
        # debug() # debug is necessary to attach the debugger to the process
        # Load the config file
        set_seed(0)
        cls.config = OmegaConf.load(cls.config_path)
        dataset_cfg = cls.config.get(cls.name,None)
        cls.data_list = instantiate(dataset_cfg)

    @classmethod
    def teardown_class(cls):
        del cls.data_list

    def test_dataset_not_empty(self):
        assert len(self.data_list) > 0, "Dataset is empty."

    def test_dataset_type(self):
        self.test_dataset_not_empty()
        sample = self.data_list[0]
        assert isinstance(sample, dict), f"Expected each sample to be a dict, but got {type(sample)} instead."
        
    def test_required_keys_exist(self):
        """
        Check that each required key exists in the dataset sample.
        """
        self.test_dataset_not_empty()  

        required_keys = ["ori_question", "question", "answer", "safe", "category"]
        image_keys = ["image", "image_path"]

        sample = self.data_list.data[0]
        sample_keys = sample.keys()

        # all required keys must exist
        for key in required_keys:
            assert key in sample_keys, f"Missing key: {key}"

        # At least one of image keys must exist
        assert any(k in sample_keys for k in image_keys), \
            f"At least one of {image_keys} must be present in sample keys: {sample_keys}"
        
    def test_sample_count(self):
        """
        Check that the dataset has the expected number of samples.
        """
        sample_count = self.config.get(self.name).get("sample", -1)
        if sample_count > 0:
            assert len(self.data_list) == sample_count, f"Expected {sample_count} samples, but got {len(self.data_list)} instead."
        
    def test_samples_per_category(self):
        """
        Check that the dataset has the expected number of samples per category.
        """
        samples_per_category = self.config.get(self.name).get("samples_per_category", -1)
        if samples_per_category > 0:
            category_count = {}
            for sample in self.data_list:
                category = sample["category"]
                if category not in category_count:
                    category_count[category] = 0
                category_count[category] += 1
            assert all(count == samples_per_category for count in category_count.values()), \
                f"Expected {samples_per_category} samples per category, but got {category_count} instead."
        

class TestFigStepDataset(BaseTestDataset):
    config_path = "configs/dataset/FigStep.yaml"



class TestMMSafetyBench(BaseTestDataset):
    config_path = "configs/dataset/MMSafetyBench.yaml"


class TestSPA_VL(BaseTestDataset):
    config = { 
        "dataset": {
            "_target_": "src.dataset.SPA_VL",
            "data_path": "data/SPA_VL/",
            "split": "test",
            "question": "question",
            "answer": "chosen",
            "image_PATH": "image_name",
            "image": "image",
            "safe": False,
            "seed": 0
        }
    }

    @classmethod
    def setup_class(cls):
        # Load the config file
        set_seed(cls.config["dataset"]["seed"])
        cls.data_list = instantiate(cls.config["dataset"])

    def test_train_dataset(self):
        _config = deepcopy(self.config)
        _config["dataset"]["split"] = "train"
        data_list = instantiate(_config["dataset"])
        # assert len(data_list) == 93258, f"Expected 93258 samples, but got {len(data_list)} instead."
        assert len(data_list) == 1000, f"Expected 93258 samples, but got {len(data_list)} instead."
        print(data_list[0])
        
    def test_val_dataset(self):
        _config = deepcopy(self.config)
        _config["dataset"]["split"] = "validation"
        data_list = instantiate(_config["dataset"])
        assert len(data_list) == 7000, f"Expected 7000 samples, but got {len(data_list)} instead."
        print(data_list[0])

    def test_test_dataset(self):    
        _config = deepcopy(self.config)
        _config["dataset"]["split"] = "test"
        data_list = instantiate(_config["dataset"])
        assert len(data_list) == 265, f"Expected 265 samples, but got {len(data_list)} instead."
        print(data_list[0])


class TestPGDAttackDataset(BaseTestDataset):
    config_path = "configs/dataset/PGDAttackDataset.yaml"


class TestSafetyAlignedDataset(BaseTestDataset):
    config_path = "configs/dataset/SafetyAlignedDataset.yaml"
    def test_safe_unsafe_count(self):
        safe_list = self.data_list["safe"]
        count_true = sum(safe_list)
        count_false = len(safe_list) - count_true
        
        assert count_true > 0, "There should be at least one safe sample."
        assert count_false > 0, "There should be at least one unsafe sample."
        print(f"Safe count (True): {count_true}")
        print(f"Unsafe count (False): {count_false}")




