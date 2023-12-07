# Antispoof with RawNet2 

Implementation is based on the [END-TO-END ANTI-SPOOFING WITH RAWNET2](https://arxiv.org/pdf/2011.01108.pdf).

The model predicts whether a speech is real or spoofed.

You can read the [original statement](https://github.com/XuMuK1/dla2023/tree/2023/hw5_as).

## Installation guide

1. Use python3.11
```shell
conda create -n fastspeech2 python=3.11 && conda activate fastspeech2
```
2. Install libraries
```shell
pip3 install -r requirements.txt
```
3. Download [ASVSpoof 2019 Dataset](https://datashare.ed.ac.uk/handle/10283/3336), [kaggle link](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset).
4. Download my RawNet2 checkpoint
```shell
python3 scripts/download_checkpoint.py
```

## Train 
First, specify a path to the data in config, then run for training
```shell
python3 train.py -c configs/train_kaggle.json
```
The final model was trained with `configs/train_kaggle.json` config.

## Test
1. Run for testing
```shell
python3 test.py
```
`test.py` include such arguments:

Results will be saved in the `test_model/results`, you can see example in this folder.

## Wandb Report

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

Equal Error Rate implementation was taken from HSE DLA course [homework](https://github.com/XuMuK1/dla2023/blob/2023/hw5_as/calculate_eer.py).

SincConv implementation was taken from DLA course in HSE [seminar](https://github.com/XuMuK1/dla2023/blob/2023/week10/antispoofing_seminar.ipynb).

