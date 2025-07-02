from .base_dataset import BaseDataset
class PGDAttackDataset(BaseDataset):
    def __init__(self, adv_img,text_dataset,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_dataset = text_dataset
        self.adv_img = adv_img
        self.data = self.load_dataset()
        
    
    def load_dataset(self):
        if "image" in self.text_dataset.data.column_names:
            data = self.text_dataset.data.map(lambda x: {"image": self.adv_img})
        else:
            data  = self.text_dataset.data.add_column("image", [self.adv_img] * len(self.text_dataset.data))
        return data
