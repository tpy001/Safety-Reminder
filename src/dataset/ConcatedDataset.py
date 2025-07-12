"""
Dataset class used for Safety-aligned training, containing both harmful and normal data.
"""
from .base_dataset import BaseDataset
from src.utils import set_seed
from loguru import logger
from datasets import Dataset
from datasets import load_dataset
from datasets import concatenate_datasets



class ConcatedDataset(BaseDataset):
    # Keys to keep from all datasets
    keep_keys = ["question", "ori_question", "safe", "category", "chosen", "image"]
    name = "ConcatedDataset"
    def __init__(self, seed=0, dataset_list=[], sample=-1, samples_per_category=-1, *args, **kwargs):
        self.seed = seed
        set_seed(self.seed)
    
        self.dataset_list = dataset_list
        
        source_dataset = []
        # Preprocess each dataset before concatenation
        processed_datasets = []
        for dataset in dataset_list:
            data = dataset.data  # Assume each `dataset` has a `.data` attribute (of type `datasets.Dataset`)

            # Ensure all required keys are present
            for key in self.keep_keys:
                if key not in data.column_names:
                    # Fill missing columns with empty string (or use None/0 depending on your case)
                    data = data.add_column(name=key, column=["" for _ in range(len(data))])

            # Remove unwanted columns
            data = data.remove_columns([k for k in data.column_names if k not in self.keep_keys])

            # Enforce column order
            data = data.map(lambda x: {k: x[k] for k in self.keep_keys})
            _source_dataset = ([dataset.name]*len(data))
            data = data.add_column(name="source_dataset", column=_source_dataset)
            
            processed_datasets.append(data)

        # Concatenate all preprocessed datasets
        self.data = concatenate_datasets(processed_datasets)

        # sample the dataset
        if sample > 0 and sample < len(self.data):
            self.data = self.sample(sample)

        if samples_per_category > 0:
            self.data = self.sample_per_category(samples_per_category)

   