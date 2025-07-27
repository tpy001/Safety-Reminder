from .base_dataset import BaseDataset
from datasets import load_dataset,Dataset
import os
from src.utils import set_seed
import numpy as np


class VLGuard(BaseDataset):
    name = "VLGuard"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def sample(self, sample_num):
        set_seed(self.seed)
        return self.data.select(range(sample_num*2))
    
    def load_dataset(self):
        dataset = load_dataset(self.data_path, data_files = self.split+".json",split='train')
        data = {
            "question":[],
            "chosen":[],
            "safe": [],
            "image":[],
        }

        for item in dataset:
            image = os.path.join( self.data_path, self.split, item['image'])
            safe = item['safe']
            for instr_resp in item['instr-resp']:
                response = instr_resp['response']
                safe = True
                if instr_resp['safe_instruction'] is not None:
                    instruct = instr_resp['safe_instruction']
                elif instr_resp['unsafe_instruction'] is not None:
                    instruct = instr_resp['unsafe_instruction']
                    safe = False

                data['question'].append(instruct)
                data['image'].append(image)
                data['chosen'].append(response)
                data['safe'].append(safe)

        return Dataset.from_dict(data)

