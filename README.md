
Migrate from [yjxiong](https://github.com/yjxiong/caffe/tree/02d361cadcb133a531f00331d7b21a268bd2aa0d) for MPI supported segmentation task. 

## Updates

- merge `interp_layer` from deeplabv2
- merge `bn_layer` from master branch(solved kshape problem)
- update `net.cpp` with explicit output log message 
- Add `indoor` example (see `train_script.sh`) which use multi-gpu train [PSPNet](https://github.com/hszhao/PSPNet)


## Compile

MPI version > 1.7.2, `--with-cuda --enable-mpi-thread-multiple` for optimal performance.

Compile with cmake is prefer for this branch(with MPI), for example:

```
# NOTE: modify openmpi install path in `cmake/Dependencies.cmake` first
mkdir build && cd build
cmake ../ -DUSE_MPI=ON -DUSE_CUDNN=OFF -DBLAS=Open
make && make install
mpirun -np ${GPU_NUM} ./install/bin/caffe train --solver=<Your Solver File> [--weights=<Pretrained caffemodel>]
```

# Notes

1. use a small learning rate to warm-up when train with mpirun BN layer
2. Memory usage about 1.4x less when trainning pspnet50 with batchsize=2

More about MPI usage refrence to [yjxiong](https://github.com/yjxiong/caffe/tree/mem).

