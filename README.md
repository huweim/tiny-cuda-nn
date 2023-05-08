# Tiny CUDA Neural Networks 

Fork from https://github.com/NVlabs/tiny-cuda-nn

Develop the MLP forward and grid forward to output the value of weights and grid tables.

# Environment

+ CUDA 11.7
+ gcc 9.4.0
+ cmake 3.24.0
+ GPU: RTX 2080 Ti

```shell
pip install tinycudann
```
# Getting Started
## Compile

```shell

cd ./bindings/torch
python setup.py install
# get the C shared library and replace the initial library
cp bindings/torch/build/lib.linux-x86_64-cpython-39/tinycudann_bindings/_75_C.cpython-39-x86_64-linux-gnu.so ~/anaconda3/envs/YOUR_ENV_NAME/lib/python3.9/site-packages/tinycudann_bindings

```

## Run


Run your TCNN project and post-process.

```shell
python train.py > input.txt
python post_process.py --print_iteration_num=1
```

`post_process.py` will create a directory named `txt`.
By default only the data of the first iteration in the file is written to file, this parameter can be set by `print_iteration_num`.
After post-processing, the files for each level of the hash table is generated, as well as the files for each level of the weights.

# Modification in this project

Compared to the [initial repo.](https://github.com/NVlabs/tiny-cuda-nn), `grid.h ` and `fully_fused_mlp.cu `are two files we modify in this project. One thread in CUDA kernel function will output the value of weights in the forward of MLP and grid tables.
```shell
|-- include
| |-- tiny-cuda-nn
| | |-- encodings
| | | |-- grid.h
|-- src
| |-- fully_fused_mlp.cu
```
