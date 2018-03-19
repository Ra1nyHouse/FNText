# FNText

## Introduction

FNText is a fast neural model for text classification.
Please note that
there are many powerful and open-source frameworks to
quickly prototype deep neural networks. It is widely used
(Google, Baidu) industry standard. However, most of them
need to translate input data into machine-readable arrays. The
input texts are padded or truncated to a fixed size. This tran
translation may increase the training time or cause information
loss. Thus we implement our model by C99 and OpemMP,
which can receive variable-length sequence as input, rather
than fixed array.


The model is shown as follows:


![model](https://raw.githubusercontent.com/Ra1nyHouse/FNText/master/model.png)

## Experiment Results

![results](https://raw.githubusercontent.com/Ra1nyHouse/FNText/master/results.png)

## Usage

### params
  * -train training data path
  * -test testing data path
  * -vali validation data path
  * -dim dimension of word vectors
  * -vocab vocabulary size
  * -epoch epoch
  * -batch-size batch size
  * -thread number of threads
  * -limit-vocab (float)  limiting vocabulary

### command line
```sh
fntext -train train/data/path -test test/data/path -vali vali/data/path -dim 600 -vocab 350000 -category 2 -epoch 5 -batch-size 1000 -thread 20 -limit-vocab 1

fntext_bi -train train/data/path -test test/data/path -vali vali/data/path -dim 600 -vocab 350000 -category 2 -epoch 5 -batch-size 1000 -thread 20 -limit-vocab 1
```
