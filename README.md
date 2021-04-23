### Hair Classifier
This

Face detector I use from facenet-pytorch
###Training
This training code for train classifier

To train model:
1. Install requirements `pip install -r requirements.txt`
2. make 2 folders with names:

    ``` hair_classifier/
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
        other code... ```
   
3. Run `python train.py`

### Infer
1. Install requirements `pip install -r requirements.txt`

2. Run `python infer.py --eval_folder ./misc/hair-val/`

   results.csv created in current directory