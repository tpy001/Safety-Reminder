from .base_dataset import BaseDataset
from datasets import load_dataset
import os
from typing import Union


class VLSafe(BaseDataset):
    def __init__(
            self,
            data_path = None, 
            data_files = None,
            split='train',
            question = 'question',
            ori_question = None,
            chosen = 'answer',
            safe = Union[str, bool],
            image = 'image',
            image_path = 'image_path',
            sample = -1 ,
            samples_per_category = -1,
            category = 'category',
            seed = 0,
            COCO_img_path = None,*args,**kwargs):
        
        self.seed = seed
        self.data_path = data_path
        self.data_files = data_files
        self.split = split

        # load the dataset
        self.data = self.load_dataset()

        # rename columns
        self.rename_column(question, 'question')
        self.rename_column(chosen, 'chosen')

        if 'question' not in self.data.column_names and 'interaction' in self.data.column_names:
            def extract_qa(example):
                question = example['interaction'][0]['user']
                answer = example['interaction'][1]['model']
                return {"question": question, "answer": answer}

            self.data = self.data.map(extract_qa)
       
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
        self.rename_column(category,'category')
        if 'category' not in self.data.column_names:
            self.data = self.data.add_column("category", ["all"] * len(self.data))
        if "chosen" not in self.data.column_names:
            self.data = self.data.add_column("chosen", [""] * len(self.data))

        # sample the dataset
        if sample > 0 and sample < len(self.data):
            self.data = self.sample(sample)

        if samples_per_category > 0:
            self.data = self.sample_per_category(samples_per_category)
        

        self.data = self.data.map(lambda image: {'image': os.path.join(COCO_img_path,image['image'])})

    

    def load_dataset(self):
        if self.data_files == "":
            return load_dataset(self.data_path, split=self.split)
        else:
            return load_dataset(self.data_path, data_files = self.data_files,split=self.split)
        
