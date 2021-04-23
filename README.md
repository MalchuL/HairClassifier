### Hair Classifier
Requirements:
```
   Python>=3.7
   
   pytorch-lightning==1.2.2
   torch==1.7.1 
   torchvision==0.8.2
   opencv-python==4.2
   ```

Face detector I use from facenet-pytorch, classifier trained from scratch
### Training
This training code for train classifier

To train model:
1. Install requirements `pip install -r requirements.txt`
2. make 2 folders with names:
```
hair_classifier/
        dataset/
            train/
                longhair/
                    imga.jpg
                    ...
                shorthair/
                    imgb.jpg
                    ...
            val/
                longhair/
                    img2.jpg
                    ...
                shorthair/
                    imgb.jpg
                    ...
        train.py
        infer.py
        other code... 
```
   
3. Run `python train.py`

### Infer
1. Install requirements `pip install -r requirements.txt`

2. Run `python infer.py --eval_folder ./misc/hair-val/`

   results.csv will be created in current directory