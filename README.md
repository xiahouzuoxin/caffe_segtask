## Introduction

This code is based on [Deeplab v2](https://bitbucket.org/aquariusjay/deeplab-public-ver2), which I merged some other features:

- Fixed the `batch_norm_layer`'s bug by replacing with [caffe-master](https://github.com/BVLC/caffe)'s `batch_norm_layer`
- Merge `crop_layer` from [caffe-master](https://github.com/BVLC/caffe)
- Merge PSPNet support by merging `bn_layer` from <https://github.com/hszhao/PSPNet>
- Merge `multi_stage_meanfield` layer from <https://github.com/bittnt/caffe/tree/crfrnn>. Some bug exist currently
- Modify `image_seg_data_layer.cpp` to support class segmentation label index > 256 (set `label_span: RG` ref ADE20K's label format)

## Install

The same to [Deeplab v2](https://bitbucket.org/aquariusjay/deeplab-public-ver2).
