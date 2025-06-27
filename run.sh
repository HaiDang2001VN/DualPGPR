#!/bin/bash

python preprocess.py --dataset cell

python train_agent.py --dataset cell

python test_agent.py --dataset cell