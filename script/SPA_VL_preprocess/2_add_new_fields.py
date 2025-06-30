"""
    Add new fields to the original dataset.
    1. Add category field to the training dataset
    2. Save the image in validation and test dataset to the disk and add a new field, image path, to the dataset
"""     

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


import os
from datasets import load_dataset
from src.utils import debug
import json
from tqdm import tqdm


def _load_dataset(data_path,split):
    if split == 'train':
        return load_dataset(data_path, split='train')
    elif split == 'test':
        return load_dataset(data_path, 'test') 
    elif split == 'validation':
        return load_dataset(data_path, 'validation')['validation']


def process_training_dataset(dataset, output_folder, meta_file_path="./data/SPA_VL/train/meta.json"):
    """
    Add category field to the training dataset
    """
    # Load the meta file of the training dataset
    meta_dataset = load_dataset('json', data_files=meta_file_path)['train']

    # convert the meta file to a list for faster access
    meta_list = list(meta_dataset)

    # Extract fields in advance for performance optimization
    get_index = lambda name: int(name.split('.')[0])

    class1, class2, class3 = [], [], []

    for data_item in tqdm(dataset, desc="Processing dataset"):
        image_name = data_item['image_name']
        index = get_index(image_name)
        meta_item = meta_list[index]

        assert meta_item['question'] == data_item['question']
        assert meta_item['chosen'] == data_item['chosen']
        assert meta_item['rejected'] == data_item['rejected']

        _class1, _class2, _class3 = meta_item["image"].split('/')[:-1]

        class1.append(_class1)
        class2.append(_class2)
        class3.append(_class3)

    new_datasest = dataset
    new_datasest = new_datasest.add_column("class1", class1)
    new_datasest = new_datasest.add_column("class2", class2)
    new_datasest = new_datasest.add_column("class3", class3)

    # Save the dataset as parquet file to the disk
    output_parquet_path =  os.path.join(output_folder, "train_converted","data.parquet")
    new_datasest.to_parquet(output_parquet_path)




def process_validation_dataset(dataset,output_folder):
    """
    Save the image to disk and add a new field, image path, to the validation dataset
    """
    # Save the image to the disk
    if not os.path.exists(os.path.join(output_folder,"val_img")):
        os.makedirs(os.path.join(output_folder,"val_img"))

    for i in range(len(dataset)):
        image = dataset[i]['image']
        image_path = os.path.join(output_folder,"val_img",f"{i}.jpg")
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(image_path)
                
    # Add the image path to the dataset
    image_path = [os.path.join(output_folder,"val_img",f"{i}.jpg") for i in range(len(dataset))]
    new_dataset = dataset.add_column("image_path",image_path)

    # Save the dataset as parquet file to the disk
    output_parquet_path =  os.path.join(output_folder, "validation_converted","data.parquet")
    new_dataset.to_parquet(output_parquet_path)


def process_test_dataset(dataset,output_folder):
    """
    Save the image to disk and add a new field, image path, to the test dataset
    """
    # Save the image to the disk
    if not os.path.exists(os.path.join(output_folder,"test_img")):
        os.makedirs(os.path.join(output_folder,"test_img"))
        os.makedirs(os.path.join(output_folder,"test_img","harm"))
        os.makedirs(os.path.join(output_folder,"test_img","help"))


    for i in range(len(dataset['harm'])):
        harm_image = dataset['harm'][i]['image']
        harm_image_path = os.path.join(output_folder,"test_img","harm",f"{i}.jpg")
        if harm_image.mode != 'RGB':
            harm_image = harm_image.convert('RGB')
        harm_image.save(harm_image_path)


    for i in range(len(dataset['help'])):
        helpful_image = dataset['help'][i]['image']
        helpful_image_path = os.path.join(output_folder,"test_img","help",f"{i}.jpg")
        if helpful_image.mode != 'RGB':
            helpful_image = helpful_image.convert('RGB')
        helpful_image.save(helpful_image_path)
                
    # Add the image path to the dataset
    harm_image_path = [os.path.join(output_folder,"test_img","harm",f"{i}.jpg") for i in range(len(dataset['harm']))]
    helpful_image_path = [os.path.join(output_folder,"test_img","help",f"{i}.jpg") for i in range(len(dataset['help']))]
    harm_dataset = dataset['harm'].add_column("image_path",harm_image_path)
    helpful_dataset = dataset['help'].add_column("image_path",helpful_image_path)

    # Save the dataset as parquet file to the disk
    output_harm_parquet_path =  os.path.join(output_folder, "test_converted","harm.parquet")
    output_help_parquet_path =  os.path.join(output_folder, "test_converted","help.parquet")

    harm_dataset.to_parquet(output_harm_parquet_path)
    helpful_dataset.to_parquet(output_help_parquet_path)

debug()

input_folder = "data/SPA_VL"
# output_folder = "data/SPA_VL_converted"
output_folder = "data/SPA_VL"


if not os.path.exists(output_folder):
    os.mkdir(output_folder)
# for split in ['train','validation','test']:
for split in ['train']:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the original dataset
    dataset = _load_dataset(input_folder,split)

    if split == "train":
        process_training_dataset(dataset,output_folder)

    elif split == "validation":
        process_validation_dataset(dataset,output_folder)
    elif split == "test":
        process_test_dataset(dataset,output_folder)

