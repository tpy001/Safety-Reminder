from  .base_dataset import BaseDataset
class JailbreakV_28K(BaseDataset):
    name = "JailbreakV_28K"
    def __init__(self,scenario = [], *args,**kwargs):
        super().__init__(*args,**kwargs)   

        self.scenario_list = scenario
        self.data = self.data.map(lambda x: {"image": os.path.join(self.data_path, x['image'])})

        #　self.data  =  self.data.filter(lambda x: x['format'] in self.scenario_list)

        # self.data = self.sample(128)