import cv2
from torchvision.datasets import ImageFolder


class MyImageDataset(ImageFolder):
    """
    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, etc) and GT image pairs.
    If only GT images are provided, generate LQ images on-the-fly.
    """

    def _find_classes(self, dir: str):
        class2idx = self.folder_name_to_class
        classes = [None for _ in class2idx.keys()]
        for k,v in class2idx.items():
            classes[v] = k
        return classes, class2idx

    def __init__(self, root: str, transforms, folder_name_to_class):
        self.folder_name_to_class = folder_name_to_class
        super().__init__(root, transform=transforms, loader=cv2.imread)


