#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(1, '../src')

from main import *

if len(sys.argv) != 7:
    print('Incorrect number of arguments')
    exit()

func_id = int(sys.argv[1])
noise_std = float(sys.argv[2])
distribution = sys.argv[3]
n_sample = int(sys.argv[4])
my_seed = int(sys.argv[5])
mod = sys.argv[6]

v_domains = ['box', 'ch', 'chplus', 'isofor']

run_one(func_id, noise_std, distribution, n_sample, my_seed, mod, v_domains)
