###
Training

make 2 folders with names:
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



Run `pip install -r requirements.txt`
Run `python infer.py --eval_folder ./hair-val/`