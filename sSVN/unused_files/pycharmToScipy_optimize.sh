#!/bin/bash

# Used to push modified library files to remote_resources
# Will allow faster debugging of external libraries with pycharm

scipy_optimize='/home/alex/anaconda3/envs/myenv/lib/python3.6/site-packages/scipy/optimize'

pycharm='/mnt/c/Users/Alex/.PyCharm2019.2/system/remote_sources/1045508341/555082905/scipy/optimize/'

# Files to forward

file1='linesearch.py'

# Copy over the files

cp $pycharm$file1 $scipy_optimize
