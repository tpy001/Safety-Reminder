from .base_dataset import BaseDataset
from datasets import load_dataset
import os
class MultiDataset(BaseDataset):
    def __init__(self, VLSafeNormal, MMVet, FigStep, MMSafetyBench, VLSafe_harmful, JailbreakDataset, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = {
            "question": [],
            "chosen": [],
            "safe": [],
            "ori_question": [],
            "image": [],
            "category": []
        }

        # # save MMVet data to given dir
        output_dir = "data/MM-Vet/images"
        # for i in range(len(MMVet)):
        #     q_id = MMVet.data['question_id'][i]
        #     image = MMVet.data['image'][i]
        #     image.save(os.path.join(output_dir,q_id + ".png"))

        image_list = [ os.path.join(output_dir,q_id + ".png")  for q_id in MMVet.data['question_id']]
        MMVet.data = MMVet.data.remove_columns("image")
        MMVet.data = MMVet.data.add_column("image", image_list)
        MMVet.data = MMVet.data.add_column("safe", [True]*len(MMVet.data))

        FigStep.data = FigStep.data.add_column("safe", [False]*len(FigStep.data))
        JailbreakDataset.data = JailbreakDataset.data.add_column("safe", [False]*len(JailbreakDataset.data))


        # 组合所有数据集
        self.dataset_list = [VLSafeNormal, MMVet, FigStep, MMSafetyBench, VLSafe_harmful, JailbreakDataset]

        for dataset in self.dataset_list:
            dataset_len = len(dataset.data["question"])  # 确保数据长度
            for key in self.data.keys():
                if key in dataset.data.column_names:
                    self.data[key].extend(dataset.data[key])  # 直接扩展列表
                else:
                    self.data[key].extend([None] * dataset_len)  # 维持长度一致

       

        self.data = Dataset.from_dict(self.data)

    def __getitem__(self, index):
        return self.data[index]
    