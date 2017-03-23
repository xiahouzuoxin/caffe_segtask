
# Usage

- Compile this branch of caffe using cmake

- Add `image_seg_data_layer.py` to `PYTHONPATH`

	```shell
	export PYTHONPATH=$PYTHONPATH:$PWD
	```

Training `pspnet50_ade20k64cls_v2` as an example. 

- Modify `solver_pspnet50_ade20k64cls_v2.prototxt`'s `device_id` to choose GPUs

- Uncomment `train_script.sh` (choose MPI process number with 4 in this example) with following lines and run it

	```shell
	nohup mpirun -np 4 ${CAFFE_ROOT}/build/tools/caffe train \
			-solver=solver_pspnet50_ade20k64cls_v2.prototxt \
			-weights=../../models/PSPNet/pspnet50_ADE20K.caffemodel &> ade20k64cls_pspnet50_v2.trainlog &
	```

# About Datasets

- NYUv2: <https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/data/nyud> with arguments `resize 321x321 & scale 1.5`
- VOC2012:  
- ADE20K:

# About 


