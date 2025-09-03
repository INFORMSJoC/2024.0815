#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(1, '../src')

from main import *

if len(sys.argv) != 2:
    print('Incorrect number of arguments')
    exit()

my_seed = int(sys.argv[1])

# Setup 'small' instances to run

func_ids = [1, 2, 3, 4, 5, 6, 7, 8]
noise_range = [0, 0.1, 0.2]
sample_range = [500, 1000, 1500]
distributions = ['normal_at_min', 'uniform']
models = ['net', 'gb', 'forest']
v_domains = ['box', 'ch', 'chplus', 'isofor', 'chp.05', 'chp.1']

###############################################################################

# Create iterations

iterations = [
    (func_id, noise_std, distribution, n_sample, mod) \
    for func_id in func_ids \
    for noise_std in noise_range \
    for distribution in distributions \
    for n_sample in sample_range \
    for mod in models
]

###############################################################################

# Do iterations

for (func_id, noise_std, distribution, n_sample, mod) in tqdm(iterations):

    run_one(func_id, noise_std, distribution, n_sample, my_seed, mod, v_domains)

###############################################################################
###############################################################################
###############################################################################

# Setup 'medium' instances to run

func_ids = [9]
noise_range = [0, 0.1, 0.2]
sample_range = [500, 1000, 1500]
distributions = ['normal_at_min']
models = ['net']
v_domains = ['box', 'ch', 'chplus', 'chp.05', 'chp.1']

###############################################################################

# Create iterations

iterations = [
    (func_id, noise_std, distribution, n_sample, mod) \
    for func_id in func_ids \
    for noise_std in noise_range \
    for distribution in distributions \
    for n_sample in sample_range \
    for mod in models
]

###############################################################################

# Do iterations

for (func_id, noise_std, distribution, n_sample, mod) in tqdm(iterations):

    run_one(func_id, noise_std, distribution, n_sample, my_seed, mod, v_domains)
