import torchvision.transforms as transforms
import albumentations as A

from albumentations.pytorch import ToTensorV2
from torchvision.transforms import Lambda
import cv2


class StagedTransform(transforms.Compose):

    def get_alb_transform(self, alb_transforms):
        composed = A.Compose(alb_transforms, p=1)
        alb_transform = [Lambda(lambda x: composed(image=x)['image'])]

        return transforms.Compose(alb_transform)

    def __init__(self, pre_transform, strong_transform=[], post_transform=[]):
        self.pre_transform = self.get_alb_transform(pre_transform)
        self.strong_transform = self.get_alb_transform(strong_transform)
        self.post_transform = self.get_alb_transform(post_transform)

        super().__init__([self.get_alb_transform(pre_transform + strong_transform + post_transform)])


def get_infer_transform(max_size=256):
    transform_list = []
    pre_process = [
        A.SmallestMaxSize(max_size, always_apply=True, interpolation=cv2.INTER_CUBIC),
        A.CenterCrop(max_size, max_size, always_apply=True)
    ]

    post_process = [A.Normalize((0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5)),
                    ToTensorV2()]

    composed = pre_process + post_process

    composed = A.Compose(composed, p=1)

    transform_list += [Lambda(lambda x: composed(image=x)['image']),
                       ]
    return transforms.Compose(transform_list)


def get_transform(opt, isTrain):
    very_rare_prob = 0.05
    rare_prob = 0.1
    medium_prob = 0.2
    normal_prob = 0.3
    often_prob = 0.6
    compression_prob = 0.35

    transform_list = []
    if isTrain:
        pre_process = [
            A.ShiftScaleRotate(shift_limit=0.01, rotate_limit=45, scale_limit=(0.6,-0.5), interpolation=cv2.INTER_CUBIC,
                               p=often_prob),
            A.SmallestMaxSize(opt.load_size, always_apply=True, interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(opt.load_size, opt.load_size, always_apply=True)]
    else:
        pre_process = [
            A.SmallestMaxSize(opt.load_size, always_apply=True),
            A.CenterCrop(opt.load_size, opt.load_size, always_apply=True)]

    strong = [

        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=normal_prob),
            A.MotionBlur(p=rare_prob),
            A.Downscale(scale_min=0.6, scale_max=0.8, interpolation=cv2.INTER_CUBIC, p=rare_prob),
        ], p=normal_prob),
        A.OneOf([A.ChannelShuffle(), A.ChannelDropout], p=medium_prob),
        A.OneOf([
            A.ToGray(p=often_prob),
            A.ToSepia(p=very_rare_prob)
        ], p=very_rare_prob),

        A.OneOf([
            A.ImageCompression(quality_lower=39, quality_upper=60, p=compression_prob),

            A.MultiplicativeNoise(multiplier=[0.92, 1.08], elementwise=True, per_channel=True, p=compression_prob),
            A.ISONoise(p=compression_prob)
        ], p=compression_prob),
        A.OneOf([
            A.CLAHE(p=normal_prob),
            A.Equalize(by_channels=False, p=normal_prob),
            A.RGBShift(p=normal_prob),
            A.HueSaturationValue(p=normal_prob),
            A.RandomBrightnessContrast(p=normal_prob),
            # A.RandomShadow(p=very_rare_prob, num_shadows_lower=1, num_shadows_upper=1,
            #               shadow_dimension=5, shadow_roi=(0, 0, 1, 0.5)),
            A.RandomGamma(p=normal_prob),
        ]),

    ]

    post_process = [A.Normalize((0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5)),
                    ToTensorV2()]

    if not isTrain:
        strong = []

    return StagedTransform(pre_process, strong, post_process)
