#!/bin/sh

python ./image_classification.py \
  --gpus $1 \
  ../CUB_200_2011/images \
  --tlf ../CUB_200_2011/train_list.txt \
  --vlf ../CUB_200_2011/test_list.txt \
  --train-batch-size 111 \
  --val-batch-size 2 \
  --resize 256,256 \
  --cropsize 224 \
  --arch vgg16 \
  --workers 4 \
  --optim-mode SGD \
  --epochs 58 \
  --start-epoch 0 \
  --lr 0.01 \
  --lr-policy multistep \
  --stepsize 30,48 \
  --gamma 0.1 \
  --print-freq 54 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --pretrained \
  --finetune


#--resume PATH
#--evaluate
#--pretrained
#--finetune

echo "Done."
