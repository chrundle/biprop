# BIPROP: BInarize-PRune OPtimizer

## Overview
**This method identifies a binary weight or binary weight and activation subnetwork within a _randomly initialized_ network that achieves performance comparable to, and sometimes better than, a weight-optimized network.** The resulting *binarized* and *pruned* networks that achieve comparable performance are called Multi-Prize Tickets, abbreviated MPTs. The experiments conducted using biprop verify the following *Multi-Prize Lottery Ticket Hypothesis*:

*A sufficiently over-parameterized neural network with random weights contains several subnetworks (winning tickets) that*
1. *Have comparable accuracy to a dense target network with learned weights (Prize 1)*
2. *Do not require any further training to achieve prize 1 (Prize 2)*
3. *Is robust to extreme forms of quantization (i.e., binary weights and/or activation) (Prize 3)*

More extensive details and motivation can be found in the following paper:

[**Multi-Prize Lottery Ticket Hypothesis: Finding Generalizable and Efficient Binary Subnetworks in a Randomly Weighted Neural Network**](https://openreview.net/forum?id=U_mat0b9iv)

## Setup

### Quick Setup
Use the `birpop.yml` file to create a conda environment in which to run biprop.

### Alternative Setup
1. Create a conda environment with python 3.7.4.
2. Use the `requirements.txt` file with ```pip install -r requirements.txt``` to install necessary requirements

## Identifying Multi-Prize Tickets (MPTs)
Configurations for various experiments can be found as YAML files in the ```configs/``` folder. Each of the configurations can be executed on a **single node** or **multi-node** (distributed) setup. It is likely that most users will require the single node configuration, the multi-node configuration was added for the larger experiments (ImageNet). Single node setups are sufficient for CIFAR-10 experiments.

The command to run a **single node** experiment is of the form:

```bash
python main.py --config <path/to/config> <override-args>
```

and a **multi-node** experiment is of the form:

```bash
python parallel_main.py --config <path/to/config> <override-args>
```

Note that the single node configuration should work as-is while the multi-node configuration will likely require minor configuration based on specific details of the user's distributed computing set up.

All ```override-args``` can be found in the `args.py` file. Examples include ```--multigpu=<gpu-ids seperated by commas, no spaces>``` to run on GPUs on a single node, and ```--prune-rate``` to set the prune rate, which denotes the fraction of weights remaining in the identified MPT. For example, a `prune_rate` of `0.4` will result in a MPT in which 40% of the weights are binarized and 60% are pruned.

### YAML Name Keys
For each model, there are configuration files for identifying Multi-Prize Tickets with binary weights and full precision activations, called MPT-1/32, and Multi-Prize Tickets with binary weights and binary activations, called MPT-1/1. Below we provide the naming conventions for configuration files corresponding to MPT-1/32 and MPT-1/1 experiments:
```
<network>_<initialization>.yml --------> MPT-1/32 (Binary weights and full precision activation)
<network>_BinAct_<initialization>.yml -> MPT-1/1  (Binary weights and binary activation)
```

Additionally, embedded in each configuration file name is the initialization used for that configuration. Below is a list of all initializations already implemented for use and the corresponding abbreviation found in the configuration file. While all of these initializations are available, note that the experiments involving MPT-1/32 and MPT-1/1 make use of the Kaiming normal and signed constant initializations.
```
(u)uc -> (unscaled) unsigned constant
(u)sc -> (unscaled) signed constant
(u)kn -> (unscaled) kaiming normal
```

### Example Run
Below is a sample call to identify a MPT-1/1 within the Conv4 network that has binarized 20% of the original weights and pruned the remaining 80%. In this call, a MPT-1/1 will be identified using two GPUs on a single node with the network weights initialized using a scaled Kaiming normal initialization:
```bash
python main.py --config configs/smallscale/conv4/conv4_BinAct_kn_unsigned.yml \
               --multigpu 0,1 \
               --name conv4_mpt_1_1 \
               --data <path/to/data-dir> \
               --prune-rate 0.2
```
To identify a MPT-1/32 within the same network, one can use the following call. Note that the only necessary change is the configuration file  but that we have also changed the name of the run so that it matches the identified MPT.
```bash
python main.py --config configs/smallscale/conv4/conv4_kn_unsigned.yml \
               --multigpu 0,1 \
               --name conv4_mpt_1_32 \
               --data <path/to/data-dir> \
               --prune-rate 0.2
```
The file `mpt_cifar_script.sh` will identify MPT-1/32 and MPT-1/1 networks in the Conv2/4/6/8 architectures on CIFAR-10 from the aforementioned paper for prune rates of 0.8, 0.6, 0.5, 0.4, 0.2, and 0.1. This script will require at least minimal modifications to run (such as providing user specific DATA and LOG directory information) but may require some additional modifications depending on the user's configuration.

### Performance on CIFAR-10 and ImageNet
Below we state the best performing MPT-1/32 and MPT-1/1 networks for CIFAR-10 and ImageNet. Note that +BN indicates subnetworks in which the batchnorm parameters were learned when identifying the subnetwork using **BIPROP**. 

| Configuration | Model | Params | % Weights Pruned | Initialization | Accuracy (CIFAR-10) |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| MPT-1/32 | VGG-Small  |  0.23M  | 95% | Kaiming Normal | 91.48% |
| MPT-1/32 | ResNet-18  |  2.6M  | 80% | Kaiming Normal | 94.66% |
| MPT-1/32 +BN | ResNet-18  |  2.6M  | 80% | Kaiming Normal | 94.8% |
| MPT-1/1 | VGG-Small (1.25 width) |  1.44M  | 75% | Kaiming Normal | 89.11% |
| MPT-1/1 +BN | VGG-Small  (1.25 width) |  1.44M  | 75% | Kaiming Normal | 91.9% |

Below we state the best performing MPT-1/32 networks for ImageNet on various networks.

| Configuration | Model | Params | % Weights Pruned | Initialization | Accuracy (ImageNet) |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| MPT-1/32 | WideResNet-50  |  13.7M  | 80% | Signed Constant | 72.67% |
| MPT-1/32 +BN | WideResNet-50  |  13.7M  | 80% | Signed Constant | 74.03% |
| MPT-1/1 | WideResNet-34  |  19.3M  | 60% | Kaiming Normal | 45.06% |
| MPT-1/1 +BN | WideResNet-34  |  19.3M  | 60% | Kaiming Normal | 52.07% |


To use a pretrained model use the ```--pretrained=<path/to/pretrained-checkpoint>``` flag. Pretrained ImageNet models are provided in the `pretrained` directory.

### Tracking

```
tensorboard --logdir runs/ --bind_all
```

When your experiment is done, a CSV entry will be written (or appended) to ```runs/results.csv```. Your experiment base directory will automatically be written to ```runs/<config-name>/prune-rate=<prune-rate>/<experiment-name>``` with ```checkpoints/``` and ```logs/``` subdirectories. If your experiment happens to match a previously created experiment base directory then an integer increment will be added to the filepath (eg. ```/0```, ```/1```, etc.). Checkpoints by default will have the first, best, and last models. To change this behavior, use the ```--save-every``` flag.


## Requirements

**BIPROP** has been tested with Python 3.7.4, CUDA 10.0/10.1 and PyTorch 1.3.0/1.3.1. Below is a list of requirements that can be used to install requirements (other than Python and CUDA):

```
absl-py==0.8.1
grpcio==1.24.3
Markdown==3.1.1
numpy==1.17.3
Pillow==6.2.1
protobuf==3.10.0
PyYAML==5.1.2
six==1.12.0
tensorboard==2.0.0
torch==1.3.0
torchvision==0.4.1
tqdm==4.36.1
Werkzeug==0.16.0
```

## License

SPDX-License-Identifier: (Apache-2.0)

LLNL-CODE-817561
