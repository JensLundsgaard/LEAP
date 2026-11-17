#!/bin/bash
# run_c_ect.sh

mkdir -p data/Letter-high

tar -xf Letter-high.tar -C data/Letter-high

# the below command is such because of how CHTC rearranges files
/opt/conda/bin/python -m experiments.train_gcn --dataset-name=Letter-high --learn-directions=True --model-name GNNclECT 


