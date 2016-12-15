CAFFE_ROOT=/mnt/disk/0/hzxiahouzuoxin/deeplab-public-ver2/

# ----------------------------------------------------------------------------------------------
#  Trainning Deeplab-LargeFOV-MSc-VOC2012
# ----------------------------------------------------------------------------------------------
# nohup ${CAFFE_ROOT}/build/tools/caffe.bin train -gpu=1 \
# 		-solver=solver.prototxt \
# 		-weights=../../models/init_models/vgg16_20M_coco.caffemodel &> voc2012.trainlog &

# ----------------------------------------------------------------------------------------------
#  Trainning Deeplab-LargeFOV-MSc-CRF-VOC2012, FIXME
# ----------------------------------------------------------------------------------------------
# nohup ${CAFFE_ROOT}/build/tools/caffe.bin train -gpu=0 \
# 		-solver=solver_crfrnn.prototxt \
#  		-weights=./snapshots/MSC-LargeFOV-voc12_iter_6000.caffemodel &> voc2012_crfrnn.trainlog &

# ----------------------------------------------------------------------------------------------
#  Trainning Deeplab-LargeFOV-MSc-NYUv2_40cls
# ----------------------------------------------------------------------------------------------
# nohup ${CAFFE_ROOT}/build/tools/caffe.bin train -gpu=0 \
# 		-solver=solver_nyuv2_40cls.prototxt \
# 		-weights=./snapshots/MSC-LargeFOV-voc12_iter_6000.caffemodel &> nyuv2_40cls.trainlog &

# ----------------------------------------------------------------------------------------------
#  Trainning PSPNet101-NYUv2_40cls
#  Max BatchSize on K40 is 1. TODO
# ----------------------------------------------------------------------------------------------
# nohup ${CAFFE_ROOT}/build/tools/caffe.bin train -gpu=0,1 \
# 		-solver=solver_pspnet101_nyuv2_40cls.prototxt \
#  		-weights=../../models/PSPNet/pspnet101_VOC2012.caffemodel &> nyuv2_40cls_pspnet101.trainlog &

# ----------------------------------------------------------------------------------------------
#  Trainning PSPNet50-NYUv2_40cls
# ----------------------------------------------------------------------------------------------
nohup ${CAFFE_ROOT}/build/tools/caffe.bin train -gpu=1 \
		-solver=solver_pspnet50_nyuv2_40cls.prototxt \
 		-weights=../../models/PSPNet/pspnet50_ADE20K.caffemodel &> nyuv2_40cls_pspnet50.trainlog &

