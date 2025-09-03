#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(1, '../src')

from main import *

# Setup instances to run

my_seeds = range(2023, 2023 + 100) # Goal 100
func_ids = [1, 2, 3, 4, 5, 6, 7, 8]
noise_range = [0, 0.5, 1.0]
sample_range = [500, 1000, 1500]
distributions = ['normal_at_min', 'uniform']
models = ['net', 'gb', 'forest']
v_domains = ['box', 'ch', 'chplus', 'isofor']

###############################################################################

# Create iterations

iterations = [
    (func_id, noise_std, distribution, n_sample, my_seed, mod) \
    for func_id in func_ids \
    for noise_std in noise_range \
    for distribution in distributions \
    for n_sample in sample_range \
    for my_seed in my_seeds \
    for mod in models
]

###############################################################################

# Do iterations

for (func_id, noise_std, distribution, n_sample, my_seed, mod) in tqdm(iterations):

    run_one(func_id, noise_std, distribution, n_sample, my_seed, mod, v_domains)
