     
from .base_dataset import BaseDataset
import json
from datasets import Dataset
import os

class MMSafetyBench(BaseDataset):
    def __init__(self, image_type='SD+TYPO',scenario=['01-Illegal_Activitiy'],sample:int = -1 ,*args, **kwargs):
        super().__init__()
        self.data_path = kwargs["data_path"]
        def load_scenario_img(scenario="01-Illegal_Activitiy", id = -1):
            if image_type == 'TYPO':
                file_path = os.path.join(self.data_path, "imgs/{scen}/TYPO/{id}.jpg".format(scen=scenario,id=id))
            elif image_type == 'SD':
                file_path = os.path.join(self.data_path, "imgs/{scen}/SD/{id}.jpg".format(scen=scenario,id=id))
            elif image_type == 'SD+TYPO':
                file_path = os.path.join(self.data_path, "imgs/{scen}/SD_TYPO/{id}.jpg".format(scen=scenario,id=id))
            return file_path
        
        self.image_type = image_type    
        self.scenario_list = scenario

        data = {
            "id": [],
            "question":[],
            "ori_question":[],
            "image":[],
            "category":[]
        }

        for scen in self.scenario_list:
            file_path =  os.path.join(self.data_path, "processed_questions/{scen}.json".format(scen=scen))
            with open(file_path) as f:
                data_list = json.load(f)
            for i in data_list:
                data['ori_question'].append(data_list[i]['Question'])
                if image_type == 'SD':
                    data['question'].append(data_list[i]['Rephrased Question(SD)'])
                else:
                    data['question'].append(data_list[i]['Rephrased Question'])
                data['image'].append(load_scenario_img(scen,i))
                data['category'].append(scen)
                data['id'].append(i)

        data['safe'] = [False] * len(data['question'])
        data['chosen'] = [""] * len(data['question'])


        self.data = Dataset.from_dict(data)

        if sample > 0 and sample < len(self.data):
            self.data = self.sample(sample)

        if "samples_per_category" in kwargs.keys():
            samples_per_category = kwargs["samples_per_category"]
            self.data = self.sample_per_category(samples_per_category)
