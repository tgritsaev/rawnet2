# Antispoofing with RawNet2 
Implementation is based on the [END-TO-END ANTI-SPOOFING WITH RAWNET2](https://arxiv.org/pdf/2011.01108.pdf) article.

The model predicts whether a speech is real or spoofed.

You can read the [original statement](https://github.com/XuMuK1/dla2023/tree/2023/hw5_as).

## Installation guide
1. Use python3.11
```shell
conda create -n rawnet2 python=3.11 && conda activate rawnet2
```
2. Install libraries
```shell
pip3 install -r requirements.txt
```
3. To reproduce training, download [ASVSpoof 2019 Dataset](https://datashare.ed.ac.uk/handle/10283/3336), [Kaggle link](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset).
4. To test the quality of my solution, download my RawNet2 checkpoint:
```shell
python3 scripts/download_checkpoint.py
```

## Train 
First, specify a path to the data in a config, then run for training
```shell
python3 train.py -c configs/train_kaggle.json
```
The final model was trained with the `configs/train_kaggle.json`.

## Test
```shell
python3 test.py
```
`test.py` include such arguments:
* `-c, --config, default="configs/train_kaggle.json"`
* `-cp, --checkpoint, default="test_model/as-checkpoint.pth"`
* `-i, --input-dir, default="test_model/audios"`

Results will be saved in the `test_model/results`.

## Wandb 

1. [Wandb report](https://wandb.ai/tgritsaev/dla5/reports/Anti-spoofing-with-RawNet2--Vmlldzo2MjMyMzQ4).
2. [Wandb project](https://wandb.ai/tgritsaev/dla5/overview?workspace=user-tgritsaev).

## Credits
This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

Equal Error Rate implementation was taken from the HSE DLA course [homework](https://github.com/XuMuK1/dla2023/blob/2023/hw5_as/calculate_eer.py). SincConv implementation was taken from the HSE DLA course [seminar](https://github.com/XuMuK1/dla2023/blob/2023/week10/antispoofing_seminar.ipynb).

