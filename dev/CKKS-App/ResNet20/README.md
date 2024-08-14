This repo contains benchmark code for resnet20
### build and run

Prerequisite:
- cmake
- g++ or clang
- OpenFHE
1. build the project

Set openfhe path in the CMakeLists.txt file and then run the following command.

```shell
mkdir build && cd build
cmake ..
make
```

1. Execute the project

```shell
./Resnet20 generate_keys 1
./Resnet20 load_keys 1 [input input_name]
```

### Performance Overview

|             | Memory Usage(GB) | Latency(s)    |
|-------------|------------|-----------------|
| ResNet-20   | 46.9  |  1212
