#!/bin/bash
# run_main_experiment.sh

mkdir -p data/Letter-high

tar -xf Letter-high.tar -C data/Letter-high

# the below command is such because of how CHTC rearranges files
/opt/conda/bin/python -m experiments.train_gcn --dataset-name=Letter-high --use-ect=True --learn-directions=True --use-pos-enc=False --ect-hops=1 

