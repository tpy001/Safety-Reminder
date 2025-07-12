from .base_dataset import BaseDataset
import numpy as np
from datasets import Dataset
class justinphan_harmful_harmless(BaseDataset):
    name = "justinphan_harmful_harmless"
    def __init__(self, paired=False,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paired = paired

        if not paired:
            questions = np.concatenate(self.data['question']).tolist()
            ori_questions = np.concatenate(self.data['ori_question']).tolist()
            flags = np.concatenate(self.data['safe']).tolist()

            new_data = {
                'question': questions,
                'ori_question': ori_questions,
                'safe': flags
            }

            self.data = Dataset.from_dict(new_data)


