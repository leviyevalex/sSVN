#!/bin/bash

# Add stein_variational_newton package to path so it runs

program='stein_variational_newton'

if grep -Rq "$program" ~/.bashrc
then
    echo ' '
    echo '-----------------------------------'
    echo 'Project root already added to path!'
    echo '-----------------------------------'
    echo ' '
else
    echo ' ' >> ~/.bashrc
    echo '# Added python path to run '"$program" >> ~/.bashrc
    echo 'export PYTHONPATH=$PYTHONPATH:'"$(pwd)" >> ~/.bashrc
    tail -3 ~/.bashrc
    echo ' '
    echo '-----------------------------'
	echo 'Added project root to path!'
    echo '-----------------------------'
    echo ' '
fi

