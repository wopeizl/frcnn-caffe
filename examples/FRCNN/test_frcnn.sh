#!/usr/bin/env sh
# This script test four voc images using faster rcnn end-to-end trained model (ZF-Model)
# determine whether $1 is empty
if [ ! -n "$1" ] ;then
    echo "$1 is empty, default is 0"
    gpu=0
else
    echo "use $1-th gpu"
    gpu=$1
fi

pid=$$

BUILD=build/examples/FRCNN/test_frcnn.bin

$BUILD --gpu $gpu \
       --model models/FRCNN/zf/test.prototxt \
       --weights models/FRCNN/VGG16_faster_rcnn_final.caffemodel \
       --default_c examples/FRCNN/config/voc_config.json \
#       --image_root VOCdevkit/VOC2007/JPEGImages/ \
#       --image_list examples/FRCNN/dataset/voc2007_test.txt \
#       --out_file examples/FRCNN/results/voc2007_test_zf_${pid}.frcnn \
       --max_per_image 100

CAL_RECALL=examples/FRCNN/calculate_voc_ap.py

python $CAL_RECALL  --gt examples/FRCNN/dataset/voc2007_test.txt \
            --answer examples/FRCNN/results/voc2007_test_zf_${pid}.frcnn \
            --diff examples/FRCNN/dataset/voc2007_test.difficult \
            --overlap 0.5
