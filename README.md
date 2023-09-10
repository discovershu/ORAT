# Outlier Robust Adversarial Training
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)

Shu Hu, Zhenhuan Yang, Xin Wang, Yiming Ying, and Siwei Lyu
_________________

This repository is the official implementation of our paper 
"Outlier Robust Adversarial Training", 
which has been accepted by **ACML 2023**. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

For AutoAttack, we download code from https://github.com/fra31/auto-attack.git and save it to the folder AA.
## How to generate real data with noise
 ```test
python utils/generate_real_data_with_noise.py
```
In generate_real_data_with_noise.py, you can change noise_ratio_value from {10, 20, 30 40} and asym from {True, False} and then
generate your specific noise data.

The name of the generated data will be "{mnist, cifar10, cifar100}\_noise\_{sym, asym}_{10,20,30,40}". 

## How to run the ORAT code

### 1. To run ORAT on MNIST
You can modify the value of k and m. However, we use k=60000 and m=1 as an example,
#### For epsilon=0.1, 
 ```test
python MNIST_ORAT.py --epsilon 0.1 --step_size 0.025 --dataset mnist --lr 0.03 --k 60000 --m 1 
```
#### For epsilon=0.2, 
 ```test
python MNIST_ORAT.py --epsilon 0.2 --step_size 0.05 --dataset mnist --lr 0.03 --k 60000 --m 1 
```

### 2. To run ORAT on CIFAR-10
You can modify the value of k and m. However, we use k=50000 and m=1 as an example,
#### For epsilon=2/255, 
 ```test
python CIFAR10_ORAT.py --epsilon 0.0078  --dataset cifar10 --lr 0.1 --k 50000 --m 1
```
#### For epsilon=8/255, 
 ```test
python CIFAR10_ORAT.py --epsilon 0.031 --step_size 0.00775  --dataset cifar10 --lr 0.1 --k 50000 --m 1
```

### 3. To run ORAT on CIFAR-100
You can modify the value of k and m. However, we use k=50000 and m=1 as an example,
#### For epsilon=2/255, 
 ```test
python CIFAR100_ORAT.py --epsilon 0.0078  --dataset cifar100 --lr 0.1 --k 50000 --m 1
```
#### For epsilon=8/255, 
 ```test
python CIFAR100_ORAT.py --epsilon 0.031 --step_size 0.00775  --dataset cifar100 --lr 0.1 --k 50000 --m 1
```

## Citation
Please kindly consider citing our paper in your publications. 
```bash
@inproceedings{hu2023outlier,
  title={Outlier Robust Adversarial Training},
  author={Hu, Shu and Yang, Zhenhuan and Wang, Xin and Ying, Yiming and Lyu, Siwei},
  booktitle={The 15th Asian Conference on Machine Learning (ACML 2023)},
  year={2022},
  organization={PMLR}

