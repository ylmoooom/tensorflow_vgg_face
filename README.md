Train VGG Face with Tensorflow on LFW dataset

1. Download LFW dataset http://vis-www.cs.umass.edu/lfw/lfw.tgz into /dataset
2. Git clone facenet https://github.com/davidsandberg/facenet, move two files from facenet/src/align/align_dataset_mtcnn.py, facenet/src/align/align_dataset.py into facenet/src
3. Run ./facenet/src/align_dataset_mtcnn.py ./dataset/lfw/ ./dataset/lfw-aligned/ --image_size 224 --margin 34
4. Run ./dataset/predata_LFW.py
5. Download model https://www.dropbox.com/sh/x5x661adl3vud5k/AACZCkCpEP1-dOIKiUJLu48la?dl=0 into ./
6. Run ./lfw-train-62/myl_vgg_main.py &
7. Run tensorboard --logdir=./lfw-train-62/train_master/ &
