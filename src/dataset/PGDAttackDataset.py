import os
from PIL import Image
from .base_dataset import BaseDataset
from datasets import Dataset

class PGDAttackDataset(BaseDataset):
    name = "PGDAttackDataset"
    def __init__(self, adv_img_folder, text_dataset, sample=-1, samples_per_category = -1,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_dataset = text_dataset
        self.adv_img_folder = adv_img_folder

        # Load all adversarial images
        self.adv_images = self.load_adv_images()

        # Create cross-paired image-text dataset
        self.data =  Dataset.from_list(self.cross_pair_dataset())

        # Optionally sample a subset
        if sample > 0 and sample < len(self.data):
            self.data = self.sample(sample)

        if samples_per_category > 0:
            self.data = self.sample_per_category(samples_per_category)

    def load_adv_images(self):
        """Load all images from the adversarial image folder"""
        image_files = sorted([
            os.path.join(self.adv_img_folder, f)
            for f in os.listdir(self.adv_img_folder)
            if f.lower().endswith((".bmp"))
        ])
        return image_files

    def cross_pair_dataset(self):
        """Create all image-text combinations"""
        paired_data = []
        for img in self.adv_images:
            for item in self.text_dataset.data:
                paired_data.append({
                    "image": img,
                    **{k: v for k, v in item.items()}
                })
        return paired_data

    def sample(self, n):
        """Randomly sample n examples from the Hugging Face Dataset"""
        return self.data.shuffle(seed=self.seed).select(range(n))
