<h1 align="center">
  <br>
  DOS
  <br>
</h1>

<h4 align="center">
  An official PyTorch implementation of ICLR 2024 paper
  <br>
  "DOS: Diverse Outlier Sampling for Out-of-Distribution Detection"
</h4>

<div align="center">
  <a href="https://arxiv.org/abs/2306.02031" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-arXiv-red?style=flat-square">
  </a> &nbsp;&nbsp;&nbsp;
  <a href=''>
    <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square">
  </a>
</div>

<p align="center">
  <a href="#get-started">Get Started</a> •
  <a href="#acknowledgements">Acknowledgements</a> •
  <a href="#citation">Citation</a>
</p>

## Get Started
### Overview
This repository is an official PyTorch implementation of ICLR 2024 paper 'DOS: Diverse Outlier Sampling for Out-of-Distribution Detection'. The illustration of our algorithm is shown as below:
![diagram](https://github.com/lygjwy/DOS/blob/main/diagram.png)

### Requirements
'''
pip install -r requirements.txt
'''

### Training
'''
python train_diverse.py
'''

### Evaluation
'''
OODs="svhn lsunc dtd places365_10k tinc lsunr tinr isun"
python detect.py --id cifar100 --ood $OODs --score abs --pretrain /path/to/trained/classifier
'''

### Results
![diagram](https://github.com/lygjwy/DOS/blob/main/result.png)

## Citation
If you find our repository useful for your research, please consider citing our paper:
'''
@inproceedings{
jiang2024dos,
title={{DOS}: Diverse Outlier Sampling for Out-of-Distribution Detection},
author={Wenyu Jiang and Hao Cheng and MingCai Chen and Chongjun Wang and Hongxin Wei},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=iriEqxFB4y}
}
'''
