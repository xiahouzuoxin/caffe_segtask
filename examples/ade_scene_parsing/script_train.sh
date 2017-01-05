CAFFE_ROOT=/mnt/disk/0/hzxiahouzuoxin/deeplab-public-ver2/

# ----------------------------------------------------------------------------------------------
#  Trainning Deeplab-LargeFOV-MSc-VOC2012
# ----------------------------------------------------------------------------------------------
nohup ${CAFFE_ROOT}/build/tools/caffe.bin train -gpu=1 \
		-solver=solver_ade20k.prototxt \
		-weights=../../models/DeepLab_init_models/vgg16_20M_coco.caffemodel &> ade20k.trainlog &

