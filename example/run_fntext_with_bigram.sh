#!/bin/sh
../fntext_bi -train ag.train.txt -test ag.test.txt -dim 400 -vocab 110000 -category 4 -epoch 10 -batch-size 500 -thread 20
