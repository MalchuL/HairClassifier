"""create dataset and dataloader"""
from data.image_dataset import MyImageDataset


def create_dataset(dataset_opt):
    return MyImageDataset(**dataset_opt)

