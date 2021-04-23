import argparse
import cv2

import numpy as np
import torch
from facenet_pytorch.models.mtcnn import MTCNN
from omegaconf import OmegaConf
from tqdm import tqdm

from pathlib import Path

from experiment import HairClassifier
from transforms.transform import get_infer_transform
from utils.infer_utils import crop_faces

SCALE = 1.3
PATH_TO_IMAGES = "/home/malchul/work/projects/hair_classifier/val_images"
FORMAT_FILES = ['.png', '.jpg', '.jpeg']

parser = argparse.ArgumentParser('Detect faces on image')
parser.add_argument('--model_path',
                    default='./pretrained/HairClassifier_04-06-12__ShuffleNetV2_lr_0.01_iters_1_bs_64/quant_model.pth')
parser.add_argument('--is_quant', action='store_false', help='if model is quantized, pass this argument')
parser.add_argument('--eval_folder', help='path to eval folder with images')
parser.add_argument('--output_data', default='./result.csv', help='path to output file')
parser.add_argument('--dump_images', default=None, help='dump images for debug')

args = parser.parse_args()

if args.dump_images:
    output_path = Path('out')

if __name__ == '__main__':

    detector = MTCNN()

    config = OmegaConf.load('configs/main_config.yml')
    print('loading quantized model')
    if args.is_quant:
        model = torch.jit.load(args.model_path)
        model.eval()
        for _ in range(10):
            model.dequant(
                model(model.quant(torch.rand(1, 3, config.datasets.train.load_size, config.datasets.train.load_size))))
    else:
        model = HairClassifier.load_from_checkpoint(args.model_path)
        model.eval()
    print('loading complete')

    images_folder = Path(args.eval_folder)
    filenames = []

    # List of videos
    for ext in FORMAT_FILES:
        filenames.extend(images_folder.rglob('*' + ext))
    filenames = sorted(list(map(str, filenames)))

    transforms = get_infer_transform(config.datasets.train.load_size)

    result = []

    for file_id, filename in tqdm(list(enumerate(filenames))):
        print(f'file:{filename}')

        frame = cv2.imread(filename)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection = detector.detect(frame)

        class_img = -1
        face_crops = crop_faces(frame, detection, SCALE)

        for crop_id, crop in enumerate(face_crops):
            if crop.shape[0] > 0 and crop.shape[1] > 0:
                resized_crop = transforms(frame)

                if args.is_quant:
                    res = model.dequant(model(model.quant(resized_crop.unsqueeze(0)))).squeeze()

                else:
                    res = model(resized_crop.unsqueeze(0)).squeeze()
                print(res)
                class_img = int(res > 0)
                if args.dump_images:
                    cv2.imwrite(str(output_path / f'file_{file_id}_res_{class_img}.png'),
                                cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                break
        result.append((filename, class_img))

    with open(args.output_data, 'w') as f:
        for path, class_id in result:
            f.write(f'{path} {class_id}\n')
