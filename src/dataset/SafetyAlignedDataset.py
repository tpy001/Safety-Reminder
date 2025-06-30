"""
Dataset class used for Safety-aligned training, containing both harmful and normal data.
"""
from .base_dataset import BaseDataset
from src.utils import set_seed
from datasets import load_dataset
import os
class SafetyAlignedDataset(BaseDataset):
    def __init__(self,harmful_data,normal_data,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.harmful_data = harmful_data
        self.normal_data = normal_data
        self.data = self.load_dataset()

    def load_dataset(self):
        # mix the harmful data and normal data
        data = {}
        data['question'] = self.harmful_data.data['question'] + self.normal_data.data['question']
        # data['safe'] = self.harmful_data.data['safe'] + self.normal_data.data['safe']
        if 'answer' in self.harmful_data.data.column_names:
            data['answer'] = self.harmful_data.data['answer'] + self.normal_data.data['answer']
        else:
            logger.warning("No answer column in harmful data. Use default refusal answer: 'I am sorry, I cannot help you with that. Beacuse it is not safe.'" )
            refusal_hint = "I'm sorry, but I can't assist with that question as it raises safety and ethical concerns."
            data['answer'] =  [refusal_hint] * len(self.harmful_data) + self.normal_data.data['answer']

        data['image'] = self.harmful_data.data['image'] + self.normal_data.data['image']

        data = Dataset.from_dict(data)

        # shuffle the dataset
        set_seed(self.seed)
        data = data.shuffle() 

        return data 