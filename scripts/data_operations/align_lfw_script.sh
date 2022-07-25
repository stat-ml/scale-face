#!/bin/bash

LFW_LANDMARKS_TXT_PATH=$1
ALIGNED_OUTPUT=$2
LFW_RAW_DATA=$3

python scripts/align_dataset.py $LFW_LANDMARKS_TXT_PATH \
$ALIGNED_OUTPUT --prefix $LFW_RAW_DATA \
--image_size 96 112