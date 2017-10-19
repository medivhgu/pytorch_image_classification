#!/bin/sh

python ./image_classification.py \
  --gpus 0,1 \
  ../CUB_200_2011/images \
  --tlf ../CUB_200_2011/train_list.txt \
  --vlf ../CUB_200_2011/test_list.txt \
  --arch vgg16 \
  --workers 4 \
  --epochs 45 \
  --start-epoch 0 \
  --train-batch-size 111 \
  --val-batch-size 2 \
  --lr 0.001 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --print-freq 54 \
  --pretrained \
  --finetune


#--resume PATH
#--evaluate
#--pretrained
#--finetune

echo "Done."
