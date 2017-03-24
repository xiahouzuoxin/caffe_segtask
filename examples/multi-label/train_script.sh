
CAFFE_ROOT=/mnt/disk/0/hzxiahouzuoxin/deeplab-public-ver2/

# nohup ${CAFFE_ROOT}/build/tools/caffe.bin train -gpu=0 \
# 		-solver=solver_googlenet.prototxt \
# 		-weights=../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel &> gameai.trainlog &

nohup ${CAFFE_ROOT}/build/tools/caffe.bin train -gpu=0 \
		-solver=solver_googlenet.prototxt \
		-weights=../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel &> gameai_v2.trainlog &
