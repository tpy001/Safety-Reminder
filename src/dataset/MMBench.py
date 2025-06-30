from .base_dataset import BaseDataset
class MMBench(BaseDataset):
    def __init__(self,instruct = '', *args, **kwargs):
        super().__init__(*args, **kwargs)
       
        def process_question(data):
            question = data['question']
            choices = []
            for j in ['A', 'B', 'C', 'D']:
                choice = data[j]
                if choice != 'nan':  
                    choices.append(f"{j}. {choice}")
            return {'question': question + "\n" + "\n".join(choices) + instruct}

        self.data = self.data.map(process_question)
      