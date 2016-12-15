CAFFE_ROOT=/mnt/disk/0/hzxiahouzuoxin/deeplab-public-ver2/

# ----------------------------------------------------------------------------------------------
#  Trainning
# ----------------------------------------------------------------------------------------------
nohup ${CAFFE_ROOT}/build/tools/caffe.bin train -gpu=1 \
		-solver=solver.prototxt \
		-weights=init_resnet101.caffemodel 2> v1.trainlog &

