## Introduction

This code is based on [Deeplab v2](https://bitbucket.org/aquariusjay/deeplab-public-ver2), which I merged some other features:

- Fixed the `batch_norm_layer`'s bug by replacing with [caffe-master](https://github.com/BVLC/caffe)'s `batch_norm_layer`
- Merge `crop_layer` from [caffe-master](https://github.com/BVLC/caffe) for FCN support 
- Merge `multi_stage_meanfield` layer from <https://github.com/bittnt/caffe/tree/crfrnn> for [CRFasRNN](https://github.com/torrvision/crfasrnn) merge 
- Modify `image_seg_data_layer.cpp` to support class segmentation label index > 256. if set `label_span: RG` in prototxt,`Index=R/10*256+G`.
- Merge [LRCN](http://jeffdonahue.com/lrcn/) support with coco caption example included.
- Merge PSPNet support by merging `bn_layer` from [PSPNet](https://github.com/hszhao/PSPNet) while without MPI-support for multi-batch train currenttly.

## Tips

- How to load source image & label image for training?

	Use python interface as FCN's or deeplab's filelist format. `examples/indoor_scene_parsing/image_seg_data_layer.py` maybe help too.

- How to fetch image to segmentation models with any size?

	Ref to FCN's python script. `examples/indoor_scene_parsing/predict.py` maybe help too.

- How to display result pretty?

	`examples/indoor_scene_parsing/viz_seg.py` maybe help.

- How to use CRFasRNN?

	`examples/indoor_scene_parsing/test_pspnet50_ade20k64cls_v2.prototxt` is an example. [CRFasRNN's README](https://github.com/torrvision/crfasrnn) may help too, but some differs, 

	```
	layer {
	  name: "splitting"
	  type: "Split"
	  bottom: "conv6_interp"
	  top: "unary"
	  top: "Q0"
	}
	layer {
	  name: "crf_inf"
	  type: "MultiStageMeanfield"
	  bottom: "unary"
	  bottom: "Q0"
	  bottom: "data"
	  top: "crf_inf"
	  param { lr_mult: 10000 }
	  param { lr_mult: 10000 }
	  param { lr_mult: 1000 }
	  multi_stage_meanfield_param {
		num_iterations: 3
		compatibility_mode: POTTS
		threshold: 2
		theta_alpha: 160
		theta_beta: 3
		theta_gamma: 3
		spatial_filter_weights_str: "3 3 2"    # Remain unsetting value = 3
		bilateral_filter_weights_str: "5 3 4"  # Remain unsetting value = 4
	  }
	}
	```
## Install

The same to [Deeplab v2](https://bitbucket.org/aquariusjay/deeplab-public-ver2).
