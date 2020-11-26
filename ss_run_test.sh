#!/usr/bin/env bash

GPU_ID=0
PHASE="test" # "test", "train"
DATASET="kitti"
NET="res101"
BATCH_SIZE=1
WORKER_NUMBER=4
SESSION=300
TEST_EPOCH=100
TEST_CHECKPOINT=175

CUDA_VISIBLE_DEVICES=${GPU_ID} python ss_test_net.py \
    --dataset ${DATASET} \
    --net ${NET} \
    --checksession ${SESSION} \
    --checkepoch ${TEST_EPOCH} \
    --checkpoint ${TEST_CHECKPOINT} \
    --anno ${PHASE} \
    --cuda --vis --web 

echo 
echo $(whoami)" : "${SESSION}_${TEST_EPOCH}_${TEST_CHECKPOINT}" Finish!"
echo "Copy output pkl file to dataset folder..."
cp \
    vis/faster_rcnn_${SESSION}_${TEST_EPOCH}_${TEST_CHECKPOINT}/detections_${PHASE}.pkl \
    data/gta5_tracking/gta_${PHASE}_detections.pkl
echo "Done!!":
