CAFFE_ROOT=../../

# ----------------------------------------------------------------------------------------------
#  Trainning Deeplab-LargeFOV-MSc-VOC2012
# ----------------------------------------------------------------------------------------------
# nohup ${CAFFE_ROOT}/build/tools/caffe train -gpu=1 \
# 		-solver=solver.prototxt \
# 		-weights=../../models/init_models/vgg16_20M_coco.caffemodel &> voc2012.trainlog &

# ----------------------------------------------------------------------------------------------
#  Trainning Deeplab-LargeFOV-MSc-CRF-VOC2012, FIXME
# ----------------------------------------------------------------------------------------------
# nohup ${CAFFE_ROOT}/build/tools/caffe train -gpu=0 \
# 		-solver=solver_crfrnn.prototxt \
#  		-weights=./snapshots/MSC-LargeFOV-voc12_iter_6000.caffemodel &> voc2012_crfrnn.trainlog &

# ----------------------------------------------------------------------------------------------
#  Trainning Deeplab-LargeFOV-MSc-NYUv2_40cls
# ----------------------------------------------------------------------------------------------
# nohup ${CAFFE_ROOT}/build/tools/caffe train -gpu=0 \
# 		-solver=solver_nyuv2_40cls.prototxt \
# 		-weights=./snapshots/MSC-LargeFOV-voc12_iter_6000.caffemodel &> nyuv2_40cls.trainlog &

# Finetune From ADE20K
# nohup ${CAFFE_ROOT}/build/tools/caffe train -gpu=1 \
# 		-solver=solver_nyuv2_40cls_v2.prototxt \
# 		-weights=../ade_scene_parsing/snapshots/MSC-LargeFOV-ADE20K_iter_90000.caffemodel &> nyuv2_40cls_v2.trainlog &


# ----------------------------------------------------------------------------------------------
#  Trainning PSPNet101-NYUv2_40cls
#  Max BatchSize on K40 is 1. TODO
# ----------------------------------------------------------------------------------------------
# nohup ${CAFFE_ROOT}/build/tools/caffe train -gpu=0,1 \
# 		-solver=solver_pspnet101_nyuv2_40cls.prototxt \
#  		-weights=../../models/PSPNet/pspnet101_VOC2012.caffemodel &> nyuv2_40cls_pspnet101.trainlog &

# ----------------------------------------------------------------------------------------------
#  Trainning PSPNet50-NYUv2_40cls
# ----------------------------------------------------------------------------------------------
# nohup ${CAFFE_ROOT}/build/tools/caffe train -gpu=1 \
# 		-solver=solver_pspnet50_nyuv2_40cls.prototxt \
#  		-weights=../../models/PSPNet/pspnet50_ADE20K.caffemodel &> nyuv2_40cls_pspnet50.trainlog &

# ----------------------------------------------------------------------------------------------
#  Trainning PSPNet50-ADE20K70cls with GPU Batch=1
# ----------------------------------------------------------------------------------------------
# nohup ${CAFFE_ROOT}/build/tools/caffe train -gpu=1 \
# 		-solver=solver_pspnet50_ade20k70cls.prototxt \
#  		-weights=../../models/PSPNet/pspnet50_ADE20K.caffemodel &> ade20k70cls_pspnet50.trainlog &

# ----------------------------------------------------------------------------------------------
#  Trainning PSPNet50-ADE20K64cls
# ----------------------------------------------------------------------------------------------
# nohup ${CAFFE_ROOT}/build/tools/caffe train -gpu=1 \
# 		-solver=solver_pspnet50_ade20k64cls.prototxt \
#  		-weights=../../models/PSPNet/pspnet50_ADE20K.caffemodel &> ade20k64cls_pspnet50.trainlog &

# ${CAFFE_ROOT}/build/tools/caffe test -gpu=1 \
# 	-model=train_pspnet50_ade20k64cls.prototxt \
# 	-weights=snapshots/pspnet50-ADE20K64cls_iter_200000.caffemodel \
# 	-iterations=4780

# ====== 63cls ======
# nohup ${CAFFE_ROOT}/build/tools/caffe train -gpu=1 \
# 		-solver=solver_pspnet50_ade20k64cls_v2.prototxt \
#  		-weights=../../models/PSPNet/pspnet50_ADE20K.caffemodel &> ade20k64cls_pspnet50_v2.trainlog &

nohup /home/hzxiahouzuoxin/usr/bin/mpirun -np 3 ${CAFFE_ROOT}/build/tools/caffe train \
		-solver=solver_pspnet50_ade20k64cls_v2.prototxt \
 		-weights=../../models/PSPNet/pspnet50_ADE20K.caffemodel &> ade20k64cls_pspnet50_v2.trainlog &

# ${CAFFE_ROOT}/build/tools/caffe test -gpu=0 \
# 	-model=train_pspnet50_ade20k64cls_v2.prototxt \
# 	-weights=snapshots/pspnet50-ADE20K64cls_v2_iter_200000.caffemodel \
# 	-iterations=4780

# ----------------------------------------------------------------------------------------------
#  Trainning PSPNet50-ADE20K64cls Multiscale Version
# ----------------------------------------------------------------------------------------------
# nohup ${CAFFE_ROOT}/build/tools/caffe train -gpu=1 \
# 		-solver=solver_pspnet50_ms.prototxt \
#  		-weights=snapshots/pspnet50-ADE20K64cls_v2_iter_200000.caffemodel &> ade20k64cls_pspnet50_ms.trainlog &

# nohup ${CAFFE_ROOT}/build/tools/caffe train -gpu=1 \
# 		-solver=solver_pspnet50_msv2.prototxt \
#  		-weights=../../models/PSPNet/pspnet50_ADE20K.caffemodel &> ade20k64cls_pspnet50_msv2.trainlog &

