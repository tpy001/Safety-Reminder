from .base_dataset import BaseDataset
from datasets import load_dataset
import os

class VLSafe(BaseDataset):
    def __init__(self,COCO_img_path,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
        self.data = self.data.map(lambda image: {'image': os.path.join(COCO_img_path,image['image'])})

        if 'question' not in self.data.column_names and 'interaction' in self.data.column_names:
            questions = [] 
            answers = []
            for i in range(len(self.data)):
                interaction = self.data['interaction'][i]
                question = interaction[0]['user']
                answer = interaction[1]['model']
                questions.append(question)
                answers.append(answer)
            self.data = self.data.add_column("question",questions)
            self.data = self.data.add_column("answer",answers)

    def load_dataset(self):
        if self.data_files == "":
            return load_dataset(self.data_path, split=self.split)
        else:
            return load_dataset(self.data_path, data_files = self.data_files,split=self.split)
        
