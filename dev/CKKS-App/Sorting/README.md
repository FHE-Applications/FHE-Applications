This repo contains source code for sorting benchmark. 

Prerequisite:
- cmake
- g++ or clang
- OpenFHE
1. build the project

Set openfhe path in the CMakeLists.txt file and then run the following command.
``` bash
mkdir build && cd build
cmake ..
make
```

Run:
```bash
cd build
./Sorting
```

2. change the hyperprameter

- Length of the input array could be easily changed by input a different number into the **Sorting** function. **Sorting** function will then perform sorting on an array with the length of your input.

- Hyperparameters for non-linear function and FHE scheme should be kept untouched unless you are familiar with FHE and openFHE.
