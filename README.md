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

Run splitfolders --ratio .8 .2 -- folder_with_images