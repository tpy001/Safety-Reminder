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
    data_size= 0
    sample_count = 10
    samples_per_category = 5
    
    @classmethod
    def setup_class(cls):
        # debug() # debug is necessary to attach the debugger to the process
        # Load the config file
        set_seed(0)
        cls.config = OmegaConf.load(cls.config_path)
        cls.data_list = instantiate(cls.config)

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

        required_keys = ["ori_question", "question", "chosen", "safe", "category"]
        image_keys = ["image", "image_path"]

        sample = self.data_list.data[0]
        sample_keys = sample.keys()

        # all required keys must exist
        for key in required_keys:
            assert key in sample_keys, f"Missing key: {key}"

        # At least one of image keys must exist
        assert any(k in sample_keys for k in image_keys), \
            f"At least one of {image_keys} must be present in sample keys: {sample_keys}"
        

    def test_datasize(self):
        _config = deepcopy(self.config)
        _config["split"] = "train"
        data_list = instantiate(_config)
        assert len(data_list) == self.data_size, f"Expected {self.data_size} samples, but got {len(data_list)} instead."


    def test_sample_count(self):
        """
        Check that the dataset has the expected number of samples.
        """
        sample_count = self.sample_count
        if len(self.data_list) > sample_count:
            config = deepcopy(self.config)
            config["sample"] = sample_count
            data_list = instantiate(config)
            assert len(data_list) == sample_count, f"Expected {sample_count} samples, but got {len(self.data_list)} instead."
        
    def test_samples_per_category(self):
        """
        Check that the dataset has the expected number of samples per category.
        """
        samples_per_category = self.samples_per_category
        config = deepcopy(self.config)
        config["samples_per_category"] = samples_per_category
        data_list = instantiate(config)

        category_count = {}
        for sample in data_list:
            category = sample["category"]
            if category not in category_count:
                category_count[category] = 0
            category_count[category] += 1
        assert all(count == samples_per_category for count in category_count.values()), \
            f"Expected {samples_per_category} samples per category, but got {category_count} instead."
        

class TestFigStepDataset(BaseTestDataset):
    config_path = "configs/dataset/FigStep.yaml"
    data_size = 350



class TestMMSafetyBench(BaseTestDataset):
    config_path = "configs/dataset/MMSafetyBench.yaml"
    data_size = 394



class TestSPA_VL(BaseTestDataset):
    config = { 
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
    train_size = 1000 
    val_size = 7000
    test_size = 265

    @classmethod
    def setup_class(cls):
        # Load the config file
        set_seed(cls.config["seed"])
        cls.data_list = instantiate(cls.config)

    def test_datasize(self):
        _config = deepcopy(self.config)
        _config["split"] = "train"
        data_list = instantiate(_config)
        assert len(data_list) == self.train_size, f"Expected {self.train_size} samples, but got {len(data_list)} instead."
        
        _config["split"] = "validation"
        data_list = instantiate(_config)
        assert len(data_list) == self.val_size, f"Expected {self.val_size} samples, but got {len(data_list)} instead."

        _config["split"] = "test"
        data_list = instantiate(_config)
        assert len(data_list) == self.test_size, f"Expected {self.test_size} samples, but got {len(data_list)} instead."

class TestVLSafeNormalDataset(BaseTestDataset):
    config_path = "configs/dataset/VLSafe_normal.yaml"
    data_size = 1000


class TestVLSafeHarmfulDataset(BaseTestDataset):
    config_path = "configs/dataset/VLSafe_harmful.yaml"
    data_size = 300

class TestPGDAttackDataset(BaseTestDataset):
    config_path = "configs/dataset/PGDAttackDataset.yaml"
    data_size = 500


class TestSafetyAlignedDataset(BaseTestDataset):
    config_path = "configs/dataset/SafetyAlignedDataset.yaml"
    data_size = 2000
    samples_per_category = 4
    def test_safe_unsafe_count(self):
        safe_list = self.data_list["safe"]
        count_true = sum(safe_list)
        count_false = len(safe_list) - count_true
        
        assert count_true > 0, "There should be at least one safe sample."
        assert count_false > 0, "There should be at least one unsafe sample."
        print(f"Safe count (True): {count_true}")
        print(f"Unsafe count (False): {count_false}")


class TestConcatDataset(BaseTestDataset):
    config_path = "configs/dataset/FigStep_VLSafe.yaml"
    data_size = 350 + 1000


class TestDPODataset(BaseTestDataset):
    config_path = "configs/dataset/DPO_Dataset.yaml"
    data_size = 512
    samples_per_category = 1

    def test_required_keys_exist(self):
        """
        Check that each required key exists in the dataset sample.
        """
        self.test_dataset_not_empty()  

        required_keys = ["ori_question", "question", "chosen", "safe", "category","rejected"]
        image_keys = ["image", "image_path"]

        sample = self.data_list.data[0]
        sample_keys = sample.keys()

        # all required keys must exist
        for key in required_keys:
            assert key in sample_keys, f"Missing key: {key}"

        # At least one of image keys must exist
        assert any(k in sample_keys for k in image_keys), \
            f"At least one of {image_keys} must be present in sample keys: {sample_keys}"