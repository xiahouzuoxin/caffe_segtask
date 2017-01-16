#!/usr/bin/env bash

# CAFFE_ROOT=../../
CAFFE_ROOT=/mnt/disk/0/hzxiahouzuoxin/caffe-recurrent/
GPU_ID=0
WEIGHTS=../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
DATA_DIR=./coco/coco/h5_data/
if [ ! -d $DATA_DIR ]; then
    echo "Data directory not found: $DATA_DIR"
    echo "First, download the COCO dataset (follow instructions in data/coco)"
    echo "Then, run ./coco/coco_to_hdf5_data.py to create the Caffe input data"
    exit 1
fi

${CAFFE_ROOT}/build/tools/caffe train \
    -solver ./lrcn_solver.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID
