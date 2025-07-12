from .base_dataset import BaseDataset
from datasets import load_dataset
import os

class MMVet(BaseDataset):
    name = "MMVet"
    def load_dataset(self):
        if self.data_files is None:
            data = load_dataset(self.data_path, split=self.split)
        else:
            data = load_dataset(self.data_path, data_files = self.data_files,split=self.split)

        image_path = []
        for i in range(len(data)):
            image_path.append(os.path.join(self.data_path, "images", data[i]["image"]))

        data = data.remove_columns("image")
        data = data.add_column("image", image_path)
        return data