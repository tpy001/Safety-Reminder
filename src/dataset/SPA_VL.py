from .base_dataset import BaseDataset
from datasets import load_dataset as huggingface_load_dataset
import os
from typing import Union
from os.path import join



class SPA_VL(BaseDataset):
    def __init__(
        self,
        data_path = None, 
        data_files = None,
        split='train',
        question = 'question',
        ori_question = None,
        chosen = 'chosen',
        safe = Union[str, bool],
        image = 'image',
        image_path = 'image_path',
        sample = -1 ,
        samples_per_category = -1,
        category = 'category',
        test_data_type = "harm",
        seed = 0,
        *args,
        **kwargs
    ):
        self.test_data_type = test_data_type 
        
        self.seed = seed
        assert data_path is not None, "data_path should not be None"
        self.data_path = data_path
        self.data_files = data_files
        self.split = split

        # load the dataset
        self.data = self.load_dataset()

        # rename columns
        self.rename_column(question, 'question')
        self.rename_column(chosen, 'chosen')
        if ori_question is None:
            self.data = self.data.add_column("ori_question", self.data["question"])
        else:
            self.rename_column(ori_question, 'ori_question')
        
        if isinstance(safe, str):
            self.rename_column(safe, 'safe')
        else:
            self.data = self.data.add_column("safe", [safe] * len(self.data))
        self.rename_column(image, 'image')
        self.rename_column(image_path, 'image_path')
        if "chosen" not in self.data.column_names:
            self.data = self.data.add_column("chosen", [""] * len(self.data))

        # The below line is too slow, comment this to speed up the loading process
        self.data = self.data.map(
              lambda x: { "category": join(x["class1"], x["class2"], x["class3"])},
            desc="Adding 'category' column"
        )

        if self.split == 'train':
             self.data = self.data.map(
                lambda x: {"image": os.path.join(self.data_path, "train_converted", "images", x["category"],x["image_name"])},
            )
             
        # sample the dataset
        if sample > 0 and sample < len(self.data):
            self.data = self.sample(sample)

        if samples_per_category > 0:
            self.data = self.sample_per_category(samples_per_category)
             
        
    def load_dataset(self):
        if self.split == 'train':
            return huggingface_load_dataset(
                "json",
                data_files={"train": os.path.join(self.data_path, "train_converted", "SPA_VL_sampled.json")}
            )["train"]
        elif self.split == 'test':
            if self.test_data_type == "harm":
                file_name = "harm.parquet"
            else:
                file_name = "help.parquet"
            return huggingface_load_dataset(
                "parquet",
                data_files={"train": os.path.join(self.data_path, "test_converted", file_name)}
            )["train"]
    
        elif self.split == 'validation':
            return huggingface_load_dataset(
                "parquet",
                data_files={"train": os.path.join(self.data_path, "validation_converted", "data.parquet")}
            )["train"]
       
           
    
