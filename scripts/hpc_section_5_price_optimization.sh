#!/bin/bash

# Job name

#$ -N section_5_price_optimization

# Log file

#$ -o ../results/log/section_5_price_optimization.log

# Combining output/error messages into one file:

#$ -j y

# Specifying the Queue

#$ -q MANSCI

# One needs to tell the queue system to use the current directory as the
# working directory

#$ -cwd

# "In this case it appears to be running out of memory. In the report,
# the maxvmem value is a proxy for the largest amount of memory in use
# at any point while the job ran, and here it's right up against the
# maximum conferred by the number of granted slots on the node which ran
# the job. The MANSCI queue only has one type of compute node (in fact,
# it contains exactly one node total) with 128G for 56 slots. Thus if
# you run the job in that queue, the amount of memory per slot won't
# vary among nodes the scheduler might use to run it, so to request
# enough slots to provide e.g. 32G there, you'd figure ~2G/slot and
# request 16 in the job script:"

#$ -pe smp 16

# "The mem_free resource is mostly useful as a constraint in combination
# with the slot request, for example if you're submitting jobs you
# know will need more memory than available on some node types in your
# destination queue(s) and want to ensure the scheduler doesn't launch
# such a job on any node whose slots are a certain size or smaller:
# https://wiki.uiowa.edu/display/hpcdocs/Argon+Cluster#ArgonCluster-JobScheduler/ResourceManager"

# "The wiki mentions mem_free is not a limit, but more pertinent is that
# it's not a guaranteed minimum and doesn't confer access to that memory
# at all:
# https://wiki.uiowa.edu/display/hpcdocs/Basic+Job+Submission#BasicJobSubmission-Memoryrequest"

# #$ -l mf=32G

# The command(s) to be executed. For efficiency, the modules should already be
# loaded in the shell when submitting the jobs.

module load stack/2022.2
module load python
module load py-numpy
module load py-pandas
module load py-matplotlib
module load py-scikit-learn
module load py-torch

python section_5_price_optimization.py
