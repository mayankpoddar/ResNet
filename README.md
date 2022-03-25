# ResNet
ECE-GY 7123 Deep Learning - Mini Project 1 - ResNet on CIFAR-10 within budget

## PreRequisistes
The following python packages are required:
* torch
* torchsummary
* numpy
* tqdm
* multiprocessing

Install them manually or use this in your python notebook:
`! pip install torch torchsummary numpy tqdm multiprocessing`

## Overview

* `project`_model.py`: This file is the core model that is used in this project.
* `DataFetcher.py`: This is used to load CIFAR10 dataset.
* `customTest.ipynb`: This file is a sample model developed initially for debugging purposes. A sample dataset of vehicles was used defined by the `urls` array.
* `test.ipynb`: This file is used for testing on test dataset. It results in final accuracy of `93.16%`.
* `train.ipynb`: This file is used for training the model on train dataset.

## Results

| Type   |      Accuracy      |
|----------|:-------------:|
| Training |  92.56% |
| Testing |    93.16%   |