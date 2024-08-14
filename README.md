# FHE-Applications
## Overview

This repository contains implementations of four CKKS (Cheon-Kim-Kim-Song) homomorphic encryption applications and one TFHE (Torus Fully Homomorphic Encryption) application.

CKKS applications:
- **HELR**: Homomorphic Encryption Logistic Regression (HELR) training on 1024 samples of the 2014 US Infant Mortality dataset with 10 features. We use the implementation [openfhe-logreg-training-examples](https://github.com/openfheorg/openfhe-logreg-training-examples).
- **ResNet20**: Homomorphic ResNet-20 inference on the CIFAR-10 dataset. We use the implementation [LowMemory ResNet-20](https://github.com/narger-ef/LowMemoryFHEResNet20) and make minor modifications.
- **LSTM**: Homomorphic encryption inference for a two-layer Recurrent Neural Network (RNN). This benchmark is referred to as LSTM in [[1]](#1) and [[2]](#2). The task is to perform sentiment classification on the IMDB dataset. The batch size is 256, the word embedding size is 128, and the RNN layer consists of 128 units.
- **Sorting**: Bitonic sorting of homomorphically encrypted data. The array length is 16,384.

TFHE applications:
- **DeepCNN-x**: This application has configurations X=20, 50, 100 and takes an 8 × 8 × 1 input size. The first layer is a 3 × 3 convolution (CONV) layer with a filter size of 2, followed by another 3 × 3 CONV layer with a filter size of 92 and a stride of 2. The next X layers are 1 × 1 CONV layers, each with a filter size of 92. The last CONV layer is a 2 × 2 CONV layer with a filter size of 16, followed by a fully connected (FC) layer with 10 neurons. This benchmark is used by Morphling [[3]](#3).

More details about these benchmarks can be found in:
- [HELR](dev/CKKS-App/HELR/README.md)
- [LSTM](dev/CKKS-App/LSTM/README.md)
- [ResNet20](dev/CKKS-App/ResNet20/README.md)
- [Sorting](dev/CKKS-App/Sorting/README.md)
- [DeepCNN](dev/TFHE-App/DeepCNN-X/README.md)

## Build ##

Prerequisites:
- cmake
- g++ or clang
- [OpenFHE](https://github.com/openfheorg/openfhe-development)

OpenFHE can be built with:
```bash
git clone https://github.com/openfheorg/openfhe-development.git
cd openfhe-development
git checkout b2869a # This is the version which we conduct experiments with
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=openfhe-build
cmake --build build -j
cmake --install build
```

All CKKS benchmarks can be built with:
```bash
# take HELR as an example
cd dev/CKKS-App/HELR
cmake -S. -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=<openfhe-development path>/openfhe-build/
cmake -S. -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=~/openfhe-development/openfhe-build/
cmake --build build -j
```


## Trace files

We provide trace files (records of homomorphic operations) in the (trace)[./trace] directory.

The format of lines in trace files is:
```
operation_name([target memory address,target level],[ciphertext argument memory address,ciphertext argument level],[ciphertext/plaintext argument memory address,ciphertext/plaintext argument level])
```

For example:
```
HADD([0x55db6c83fa50,1],[0x55db956e65c0,1],[0x55db956f50a0,1])
```
`HADD` is the homomorphic operation name. `[0x55db6c83fa50,1]` represents the memory address and level of the target ciphertext. `[0x55db956e65c0,1]` and `[0x55db956f50a0,1]` represent the two arguments.

If one argument is a plaintext scalar, then the memory address and the level are represented as `-`, for example:
```
PMULT([0x557a1d10bc90,12],[0x557ca6e0c4e0,11],[-,-])
```
If one argument is a plaintext vector, then the level is represented as `-`, for example:
```
PMULT([0x557a1d10bc90,12],[0x557ca6e0c4e0,11],[0x55775586ecc0,-])
```

Note that the level here starts from 0 (fresh ciphhertext) and increases by 1 after a homomorphic multiplcation.
For Bootstrapping, its trace is recorded between `BOOTSTRAPBEGIN` and `BOOTSTRAPEND`.
```
BOOTSTRAPBEGIN([0x557ca6e11eb0,11],[0x557ca6e11eb0,11])
BOOTSTRAPEND([0x557ca6e0c4e0,11],[0x557ca6e0c4e0,11])
```

## References ##

<a id="1">[1]</a> N. Samardzic et al., “CraterLake: a hardware accelerator for efficient unbounded computation on encrypted data,” in Proceedings of the 49th Annual International Symposium on Computer Architecture, in ISCA ’22. New York, NY, USA: Association for Computing Machinery, Jun. 2022, pp. 173–187. doi: 10.1145/3470496.3527393.
<a id="2">[2]</a> S. Fan, Z. Wang, W. Xu, R. Hou, D. Meng, and M. Zhang, “TensorFHE: Achieving Practical Computation on Encrypted Data Using GPGPU,” in 2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA), Feb. 2023, pp. 922–934. doi: 10.1109/HPCA56546.2023.10071017.
<a id="3">[3]</a> Prasetiyo, A. Putra, and J.-Y. Kim, “Morphling: A Throughput-Maximized TFHE-based Accelerator using Transform-domain Reuse,” in 2024 IEEE International Symposium on High-Performance Computer Architecture (HPCA), Mar. 2024, pp. 249–262. doi: 10.1109/HPCA57654.2024.00028.
