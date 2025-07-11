"""
Dataset class for DPO training. Requires pairs of (chosen, rejected) answers for the same prompt.
"""
from .base_dataset import BaseDataset
from loguru import logger
from datasets import Dataset, concatenate_datasets

class DPO_Dataset(BaseDataset):
    def __init__(self, safety_data, utility_data, seed=0,   sample = -1 ,samples_per_category = -1,*args, **kwargs):
        self.seed = seed
        self.safety_data = safety_data.data
        self.utility_data = utility_data.data
        self.data = self.process_data()

        # sample the dataset
        if sample > 0 and sample < len(self.data):
            self.data = self.sample(sample)

        if samples_per_category > 0:
            self.data = self.sample_per_category(samples_per_category)

    def process_data(self):  
        self.utility_data = self.utility_data.add_column("rejected", ["I'm sorry, but I cannot help with that."] * len(self.utility_data))
        
        common_keys = list(set(self.safety_data.column_names) & set(self.utility_data.column_names))
        self.safety_data = self.safety_data.select_columns(common_keys)
        self.utility_data = self.utility_data.select_columns(common_keys)
      
        return concatenate_datasets([self.safety_data, self.utility_data]).shuffle(self.seed) 
    

    

 