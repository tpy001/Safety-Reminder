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
    def __init__(self,seed = 0, dataset_list = [],  sample = -1 ,samples_per_category = -1,*args,**kwargs):
        self.seed = seed
        set_seed(self.seed)
        data_list = [ dataset.data for dataset in dataset_list]
        self.data = concatenate_datasets(data_list)

        # sample the dataset
        if sample > 0 and sample < len(self.data):
            self.data = self.sample(sample)

        if samples_per_category > 0:
            self.data = self.sample_per_category(samples_per_category)

   