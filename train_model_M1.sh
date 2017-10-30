#!/bin/sh

python ./image_classification_VGG16_M1.py \
  --gpus $1 \
  ../CUB_200_2011/images \
  --tlf ../CUB_200_2011/train_list.txt \
  --vlf ../CUB_200_2011/test_list.txt \
  --train-batch-size 54 \
  --val-batch-size 36 \
  --resize 640,640 \
  --cropsize 448 \
  --arch $2 \
  --workers 8 \
  --optim-mode SGD \
  --epochs 28 \
  --start-epoch 0 \
  --lr 0.01 \
  --lr-policy multistep \
  --stepsize 12,20,26 \
  --gamma 0.1 \
  --print-freq 37 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --pretrained \
  --finetune \
  --snapshot-prefix test_CUB2011 \
  #--resume ./model_best.pth.tar \
  #--evaluate


#--resume PATH
#--evaluate
#--pretrained
#--finetune

echo "Done."
