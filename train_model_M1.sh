#!/bin/sh

python ./image_classification.py \
  --gpus $1 \
  ../CUB_200_2011/images \
  --tlf ../CUB_200_2011/train_list.txt \
  --vlf ../CUB_200_2011/test_list.txt \
  --train-batch-size 54 \
  --val-batch-size 2 \
  --resize 640,640 \
  --cropsize 448 \
  --arch vgg16 \
  --workers 4 \
  --optim-mode SGD
  --epochs 120 \
  --start-epoch 0 \
  --lr 0.002 \
  --lr-policy step \
  --stepsize 3 \
  --gamma 0.92 \
  --print-freq 111 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --pretrained \
  --finetune


#--resume PATH
#--evaluate
#--pretrained
#--finetune

echo "Done."
