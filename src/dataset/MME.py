from base_dataset import BaseDataset
import numpy as np
from src.utils import set_seed

class MME(BaseDataset):
    name = "MME"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
    def sample(self, sample_num):
        set_seed(self.seed)
        sample_index = np.random.choice(len(self.data)//2, sample_num // 2, replace=False)
        index = []
        for i in sample_index:
            index.append(2*i)
            index.append(2*i+1)
        return self.data.select(index)
