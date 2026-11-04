#!/bin/bash
# run_main_experiment.sh

mkdir -p data/Letter-high

tar -xf Letter-high.tar -C data/Letter-high

python -m experiments.train_gcn --dataset-name=Letter-high --use-ect=True --learn-directions=True --use-pe=False --ect-hops=1
