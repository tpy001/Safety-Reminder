"""
Base Dataset class for different datasets, including MMBench, MME, VLGuard, FigStep, and justinphan_harmful_harmless.
Unified Dataset Structure:
{
    "ori_question": [],   # Original question
    "question": [],       # question may after rephrasing or reformulation
    "chosen": [],         # expected chosen to the question
    "safe": [],           # Safety label (e.g., True for safe, False for unsafe)
    "image": [],          # Associated image
    "image_path": [],     # Associated image path
    "category": []        # Category or type of the question (e.g., illegal activity, self-harm, etc.)
}

"""

from typing import Union
from datasets import Dataset,load_dataset,concatenate_datasets
import numpy as np
from src.utils import set_seed
import pandas as pd


class BaseDataset():
    def __init__(self, 
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
            seed = 0,
            *args,
            **kwargs
    ):
        """
        Args:
            data_path (str, optional): Path to the dataset root directory. If None, dataset is not loaded.
            data_files (str or list, optional): Specific file(s) to load within the data_path.
            split (str): Dataset split to use. Default is 'train'.
            question (str): Name of the question field in the raw data. Will be renamed to 'question'.
            ori_question (str): Name of the original question field. Will be renamed to 'ori_question'.
                                If identical to `question`, it will be copied.
            chosen (str): Name of the chosen field in the raw data. Will be renamed to 'chosen'.
            safe (str or bool): Name of the safety label column, or the value to assign if no column exists.
                                Will be renamed to 'safe'.
            image (str): Name of the image column in the raw data. Will be renamed to 'image'.
            image_path (str): Name of the image path column in the raw data. Will be renamed to 'image_path'.
            sample (int): If > 0, randomly sample this number of examples from the dataset.
                        If -1, use the full dataset. Default is -1.
            sample_per_category (int): If > 0, randomly sample this number of examples per category.
                        If -1, use the full dataset. Default is -1.
            category (str): Name of the category column. Will be renamed to 'category'.
                            If not present, a default value 'all' will be assigned to all examples.
            seed (int): Random seed used for sampling. Default is 0.
        """
        self.seed = seed
        if data_path is not None:
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
            
    def sample_per_category(self, samples_per_category=20, category_col="category"):
        """
        Sample the same number of examples per category and combine the results.

        Args:
            samples_per_category (int): Number of samples to draw from each category.
            category_col (str): Name of the category column.

        Returns:
            The sampled dataset with the same type as self.data.
        """
        set_seed(self.seed)
        data = self.data

        sampled_parts = []
        if isinstance(data, Dataset):
            categories = list(set(data[category_col]))
        elif isinstance(data, pd.DataFrame):
            categories = data[category_col].unique()
        else:
            raise ValueError(f"Unsupported data type {type(data)}")

        for cat in categories:
            if isinstance(data, pd.DataFrame):
                subset = data[data[category_col] == cat]
                if len(subset) < samples_per_category:
                    print(f"Warning: category '{cat}' only has {len(subset)} samples, less than requested {samples_per_category}. Using all available samples.")
                    sampled_subset = subset
                else:
                    sampled_subset = subset.sample(n=samples_per_category, random_state=self.seed)
                sampled_parts.append(sampled_subset)
            else:  # HuggingFace Dataset
                subset = data.filter(lambda x: x[category_col] == cat)
                if len(subset) < samples_per_category:
                    print(f"Warning: category '{cat}' only has {len(subset)} samples, less than requested {samples_per_category}. Using all available samples.")
                    sampled_subset = subset
                else:
                    sampled_subset = subset.shuffle(seed=self.seed).select(range(samples_per_category))
                sampled_parts.append(sampled_subset)

        if isinstance(data, pd.DataFrame):
            return pd.concat(sampled_parts).reset_index(drop=True)
        else:
            return concatenate_datasets(sampled_parts)


    def sample(self, sample_num):
        set_seed(self.seed)
        sample_index = np.random.choice(len(self.data), sample_num, replace=False)
        return self.data.select(sample_index)
    
    def __len__(self):
        return len(self.data)

    def load_dataset(self):
        if self.data_files is None:
            return load_dataset(self.data_path, split=self.split)
        else:
            return load_dataset(self.data_path, data_files = self.data_files,split=self.split)
    
    def rename_column(self, old_name, new_name):
        if old_name != new_name:
            if old_name in self.data.column_names:
                self.data = self.data.rename_column(old_name, new_name)

    def __getitem__(self, index):
        return self.data[index]
    
    def keys(self):
        return self.data.column_names