from .base_dataset import BaseDataset
from datasets import load_dataset as huggingface_load_dataset
import os
from os.path import join

class SPA_VL(BaseDataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.data = self.data.remove_columns(['category'])
        # The below line is too slow, comment this to speed up the loading process
        self.data = self.data.map(
            lambda x: {
                "category": join(x["class1"], x["class2"], x["class3"])
            },
            desc="Adding 'category' column"
        )

        if self.split == 'train':
             self.data = self.data.map(
                lambda x: {"image": os.path.join(self.data_path, "train_converted", "images", x["category"],x["image"])},
            )
             
        
    def load_dataset(self):
        if self.split == 'train':
            return huggingface_load_dataset(
                "json",
                data_files={"train": os.path.join(self.data_path, "train_converted", "SPA_VL_sampled.json")}
            )["train"]
        elif self.split == 'test':
            return huggingface_load_dataset(
                "parquet",
                data_files={"train": os.path.join(self.data_path, "test_converted", "harm.parquet")}
            )["train"]
    
        elif self.split == 'validation':
            return huggingface_load_dataset(
                "parquet",
                data_files={"train": os.path.join(self.data_path, "validation_converted", "data.parquet")}
            )["train"]
       
           
    
