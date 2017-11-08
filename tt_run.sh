#!/bin/sh

./train_model_M1.sh 0,1,2,3  VGG16_M0 3 2>&1 | tee log3-2_optimal_lr0.01_step3_VGG16_M0_NRSized_NoInitial.txt

./train_model_M1.sh 0,1,2,3  VGG16_M0 2 2>&1 | tee log3-2_optimal_lr0.01_step2_VGG16_M0_WRSized_NoInitial.txt

./train_model_M1.sh 0,1,2,3  VGG16_M1 2 2>&1 | tee log3-3_optimal_lr0.01_step2_VGG16_M1_WRSized_WithInitial.txt


