"""
Dataset class used for Safety-aligned training, containing both harmful and normal data.
"""
from .base_dataset import BaseDataset
from src.utils import set_seed
from loguru import logger
from datasets import Dataset
from datasets import load_dataset
from datasets import concatenate_datasets



class SafetyAlignedDataset(BaseDataset):
    def __init__(self,harmful_data,normal_data,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.harmful_data = harmful_data
        self.normal_data = normal_data
        self.data = self.load_dataset()

    def load_dataset(self):
        harmful_ds = self.harmful_data.data
        normal_ds = self.normal_data.data

        # Handle the case where harmful_ds lacks the 'answer' column
        if 'answer' not in harmful_ds.column_names:
            logger.warning("No answer column in harmful data. Using default refusal answer.")
            refusal_hint = "I'm sorry, but I can't assist with that question as it raises safety and ethical concerns."
            harmful_ds = harmful_ds.add_column("answer", [refusal_hint] * len(harmful_ds))

        # Remove irrelevant columns from both datasets
        # For example, keep only 'question', 'answer', 'image', 'ori_question', 'safe'
        columns_to_keep = ["question", "answer", "image", "ori_question", "safe","category"]

        harmful_ds = harmful_ds.remove_columns([col for col in harmful_ds.column_names if col not in columns_to_keep])
        normal_ds = normal_ds.remove_columns([col for col in normal_ds.column_names if col not in columns_to_keep])

        # Concatenate the two datasets
        data = concatenate_datasets([harmful_ds, normal_ds])

        # Shuffle the combined dataset
        data = data.shuffle()

        return data