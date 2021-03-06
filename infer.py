import argparse
import cv2

import numpy as np
import torch
from facenet_pytorch.models.mtcnn import MTCNN
from omegaconf import OmegaConf
from tqdm import tqdm
import time
from pathlib import Path

from experiment import HairClassifier
from transforms.transform import get_infer_transform
from utils.infer_utils import crop_faces

SCALE = [1.2, 1.25, 1.3, 1.5]  # TTA


MIN_CROP_SIZE = 80
PATH_TO_IMAGES = "/home/malchul/work/projects/hair_classifier/val_images"
FORMAT_FILES = ['.png', '.jpg', '.jpeg']

parser = argparse.ArgumentParser('Detect faces on image')
parser.add_argument('--model_path',
                    default='./pretrained/shufflenetv2_epoch_94_f1_score=0.973.ckpt')
parser.add_argument('--is_quant', action='store_true', help='if model is quantized, pass this argument')
parser.add_argument('--eval_folder', help='path to eval folder with images')
parser.add_argument('--output_data', default='./result.csv', help='path to output file')
parser.add_argument('--dump_images', default=None, help='dump images for debug')
parser.add_argument('--is_cpu', action='store_true')

args = parser.parse_args()

if args.dump_images:
    output_path = Path(args.dump_images)
    output_path.mkdir(exist_ok=True)

if __name__ == '__main__':

    detector = MTCNN()

    config = OmegaConf.load('configs/main_config.yml')
    num_classes = config.model.num_classes
    print('loading model')
    if not args.is_quant:
        model = HairClassifier.load_from_checkpoint(args.model_path, strict=False)
        model.eval()
        if not args.is_cpu:
            model = model.cuda()
    else:
        model = torch.jit.load(args.model_path)
        model.eval()
        for _ in range(10):
            model.dequant(
                model(model.quant(torch.rand(1, 3, config.datasets.train.load_size, config.datasets.train.load_size))))


    print('loading complete')

    images_folder = Path(args.eval_folder)
    filenames = []


    for ext in FORMAT_FILES:
        filenames.extend(images_folder.rglob('*' + ext))
    filenames = sorted(list(map(str, filenames)))

    transforms = get_infer_transform(config.datasets.train.load_size)

    result = []

    for file_id, filename in tqdm(list(enumerate(filenames))):

        frame = cv2.imread(filename)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection = detector.detect(frame)

        class_img = -1
        face_crops = crop_faces(frame, detection, SCALE)

        infer_time = -1
        for crop_id, crops in enumerate(face_crops):
            #if crop.shape[0] > MIN_CROP_SIZE and crop.shape[1] > MIN_CROP_SIZE:

                resized_crop = torch.stack([transforms(crop) for crop in crops])

                infer_time = time.time()
                with torch.no_grad():
                    if not args.is_quant:
                        if not args.is_cpu:
                            resized_crop = resized_crop.cuda()
                        res = model(resized_crop)
                    else:
                        res = model.dequant(model(model.quant(resized_crop)))
                infer_time = time.time() - infer_time

                res = res.detach().cpu()

                # TTA aggregation
                res, _ = res.max(0)

                if num_classes == 1:
                    class_img = int(res > 0)
                else:
                    class_img = int(torch.argmax(res, 0))
                if args.dump_images:
                    crop = (((resized_crop[-1].permute(1,2,0) + 1) / 2) * 255).cpu().numpy().astype(np.uint8)
                    cv2.imwrite(str(output_path / f'class_{class_img}_file_{file_id}.png'),
                                cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                break
        print('result', (filename, class_img), 'classifier time', infer_time, 'ms')
        result.append((filename, class_img))

    with open(args.output_data, 'w') as f:
        for path, class_id in result:
            f.write(f'{path},{class_id}\n')
