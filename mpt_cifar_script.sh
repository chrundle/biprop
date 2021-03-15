# Set prune rates for tests 
#declare -a prunerates=("0.8" "0.6" "0.5" "0.4" "0.2" "0.1")
declare -a prunerates=("0.4")
# Set conv network sizes for tests 
#declare -a convsizes=("2" "4" "6" "8")
declare -a convsizes=("2")
# Set CIFAR-10 dataset directory (USER DEPENDENT)
BIPROP_DATA_DIR="/g/g14/chrundle/datasets/cifar-10-batches-py"
# Set log directory (USER DEPENDENT)
BIPROP_LOG_DIR="/p/gpfs1/chrundle/biprop-tests/cifar10-tests"

#######################################
# MPT - 1/32 : Binary weight networks #
#######################################

# Loop over prune rates for tests
for i in "${prunerates[@]}"
do
  # Run test for each conv network
  for j in "${convsizes[@]}"
  do
    bsub -nnodes 1 -W 120 python main.py --config configs/smallscale/conv$j/conv$j\_kn_unsigned.yml \
                   --epochs 250 \
                   --multigpu 0 \
                   --log-dir $BIPROP_LOG_DIR \
                   --name mpt-1-32 \
                   --data $BIPROP_DATA_DIR \
                   --prune-rate $i
  done
done

#####################################################
# MPT - 1/1 : Binary weight and activation networks #
#####################################################

# Loop over prune rates for tests
for i in "${prunerates[@]}"
do
  # Run test for each conv network
  for j in "${convsizes[@]}"
  do
    bsub -nnodes 1 -W 120 python main.py --config configs/smallscale/conv$j/conv$j\_BinAct_kn_unsigned.yml \
                   --epochs 250 \
                   --multigpu 0 \
                   --log-dir $BIPROP_LOG_DIR \
                   --name mpt-1-1 \
                   --data $BIPROP_DATA_DIR \
                   --prune-rate $i
  done
done
