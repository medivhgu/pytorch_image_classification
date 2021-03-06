#!/bin/sh

python ./image_classification_VGG16_M1.py \
  --gpus $1 \
  ../CUB_200_2011/images \
  --tlf ../CUB_200_2011/train_list.txt \
  --vlf ../CUB_200_2011/test_list.txt \
  --train-batch-size 111 \
  --val-batch-size 36 \
  --resize 256,256 \
  --cropsize 224 \
  --arch vgg16 \
  --workers 4 \
  --optim-mode SGD \
  --epochs 40 \
  --start-epoch 0 \
  --lr 0.01 \
  --lr-policy multistep \
  --stepsize 22,38 \
  --gamma 0.1 \
  --print-freq 54 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --pretrained \
  --finetune \
  --snapshot-prefix vgg16_0-22-38-40_lr0.01_bs111 \
  #--resume ./model_best.pth.tar \
  #--evaluate


#--resume PATH
#--evaluate
#--pretrained
#--finetune

echo "Done."
