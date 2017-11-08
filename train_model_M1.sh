#!/bin/sh

python ./image_classification_VGG16_M1.py \
  --gpus $1 \
  ../CUB_200_2011/images \
  --tlf ../CUB_200_2011/train_list.txt \
  --vlf ../CUB_200_2011/test_list.txt \
  --train-batch-size 54 \
  --val-batch-size 36 \
  --resize 512,512 \
  --cropsize 448 \
  --arch $2 \
  --workers 8 \
  --optim-mode SGD \
  --epochs 120 \
  --start-epoch 0 \
  --lr 0.01 \
  --lr-policy step \
  --stepsize $3 \
  --gamma 0.92 \
  --print-freq 37 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --pretrained \
  --finetune \
  --snapshot-prefix CUB200_$2_step$3 \
  #--resume ./model_best.pth.tar \
  #--evaluate


#--resume PATH
#--evaluate
#--pretrained
#--finetune

echo "Done."
