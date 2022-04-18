#!/bin/bash

# Used to push modified library files to remote_resources
# Will allow faster debugging of external libraries with pycharm

bilby='/home/alex/anaconda3/envs/myenv/lib/python3.6/site-packages/bilby/gw/'

pycharm='/mnt/c/Users/Alex/.PyCharm2019.1/system/remote_sources/1045508341/555082905/bilby/gw/'

# Files to forward

file1='utils.py'

# Copy over the files

cp $pycharm$file1 $bilby
