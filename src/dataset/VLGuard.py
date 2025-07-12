from .base_dataset import BaseDataset
from datasets import load_dataset
class VLGuard(BaseDataset):
    name = "VLGuard"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_dataset(self):
        dataset = load_dataset(self.data_path, data_files = self.split+".json",split='train')
        data = {
            "question":[],
            "answer":[],
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
                data['answer'].append(response)
                data['safe'].append(safe)

        return Dataset.from_dict(data)

