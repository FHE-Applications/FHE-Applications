# DeepCNN-X

## Preliminary

```bash
pip install -U pip wheel setuptools
pip install concrete-ml
```

## Run

```bash
python3 deepcnn-x.py # x=20
# or you can set x, i.e., number of 1-conv
python3 deepcnn-x.py 50
python3 deepcnn-x.py 100
```
## TFHE parameter

Security level in Concrete-ML is 128 bit. Currently, the security level cannot be set in Concrete-ML.

| parameter | value  |
|-----|----|
|N|512|
|n|722|
|k|2|
|$l_b$|3|

## Execution time

|x|latency (s)|
|-----|----|
|  20 | 163.33796032506507 |
|  50 | 298.02986129000783 |
| 100 | 479.98481012904085 |