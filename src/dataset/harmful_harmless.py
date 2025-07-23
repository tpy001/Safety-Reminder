from .base_dataset import BaseDataset
import numpy as np
from datasets import Dataset
class harmful_harmless(BaseDataset):
    name = "harmful_harmless"
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)


