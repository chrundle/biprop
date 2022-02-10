import argparse
import sys
import yaml
import os
import torch

from configs import parser as _parser

args = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

    # General Config
    parser.add_argument(
        "--data", help="path to dataset base directory", default="/mnt/disk1/datasets"
    )
    parser.add_argument(
        "--results", help="result filepath", default="runs/indiv_results4.csv"
    )
    parser.add_argument("--optimizer", help="Which optimizer to use", default="sgd")
    parser.add_argument("--set", help="name of dataset", type=str, default="ImageNet")
    parser.add_argument(
        "-a", "--arch", metavar="ARCH", default="ResNet18", help="model architecture"
    )
    parser.add_argument(
        "--config", help="Config file to use (see configs dir)", default=None
    )
    parser.add_argument(
        "--log-dir", help="Where to save the runs. If None use ./runs", default=None
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=20,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 20)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=None,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--warmup_length", default=0, type=int, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--bn_weight_init",
        default=None,
        type=float,
        metavar="BW",
        help="initial bn weight",
    )
    parser.add_argument(
        "--bn_bias_init",
        default=None,
        type=float,
        metavar="BB",
        help="initial bn bias",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument("--num-classes", default=10, type=int)
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        default=None,
        type=str,
        help="use pre-trained model",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--multigpu",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="Which GPUs to use for multigpu training",
    )
    parser.add_argument(
        '--world-size', 
        default=-1, 
        type=int,
        help='number of nodes for distributed training'
    )
    parser.add_argument(
        '--rank', 
        default=-1, 
        type=int,
        help='node rank for distributed training'
    )
    parser.add_argument(
        "--histograms",
        dest="histograms",
        action="store_true",
        help="write scores and score gradient histograms to tensorboard",
    )

    # Learning Rate Policy Specific
    parser.add_argument(
        "--lr-policy", default="constant_lr", help="Policy for the learning rate."
    )
    parser.add_argument(
        "--multistep-lr-adjust", default=20, type=int, help="Interval to drop lr"
    )
    parser.add_argument(
        "--lr-adjust", default=20, type=int, help="Interval to drop lr"
    )
    parser.add_argument(
        "--multistep-lr-gamma", default=0.1, type=int, help="Multistep multiplier"
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="Multistep multiplier"
    )
    parser.add_argument(
        "--name", default=None, type=str, help="Experiment name to append to filepath"
    )
    parser.add_argument(
        "--save_every", default=-1, type=int, help="Save every ___ epochs"
    )
    parser.add_argument(
        "--prune-rate",
        default=0.0,
        help="Amount of pruning to do during sparse training",
        type=float,
    )
    parser.add_argument(
        "--low-data", default=1, help="Amount of data to use", type=float
    )
    parser.add_argument(
        "--width-mult",
        default=1.0,
        help="How much to vary the width of the network.",
        type=float,
    )
    parser.add_argument(
        "--nesterov",
        default=False,
        action="store_true",
        help="Whether or not to use nesterov for SGD",
    )
    parser.add_argument(
        "--random-subnet",
        action="store_true",
        help="Whether or not to use a random subnet when fine tuning for lottery experiments",
    )
    parser.add_argument(
        "--one-batch",
        action="store_true",
        help="One batch train set for debugging purposes (test overfitting)",
    )
    parser.add_argument(
        "--conv-type", type=str, default=None, help="What kind of sparsity to use"
    )
    parser.add_argument(
        "--freeze-weights",
        action="store_true",
        help="Whether or not to train only subnet (this freezes weights)",
    )
    parser.add_argument(
        "--learn_batchnorm",
        action="store_true",
        help="Whether or not to learn batchnorm weight and bias",
    )
    parser.add_argument(
        "--tune_batchnorm",
        action="store_true",
        help="Freeze subnet, only tune batchnorm",
    )
    parser.add_argument(
        "--bn_bias_only",
        action="store_true",
        help="Whether or not to train batchnorm bias only",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether or not to print weight distributions for debugging purposes",
    )
    parser.add_argument(
        "--grad-clip",
        action="store_true",
        help="Whether or not to clip gradients",
    )
    parser.add_argument("--mode", default="fan_in", help="Weight initialization mode")
    parser.add_argument(
        "--nonlinearity", default="relu", help="Nonlinearity used by initialization"
    )
    parser.add_argument("--bn-type", default=None, help="BatchNorm type")
    parser.add_argument(
        "--init", default="kaiming_normal", help="Weight initialization modifications"
    )
    parser.add_argument(
        "--no-bn-decay", action="store_true", default=False, help="No batchnorm decay"
    )
    parser.add_argument(
        "--scale-fan", action="store_true", default=False, help="scale fan"
    )
    parser.add_argument(
        "--first-layer-dense", action="store_true", help="First layer dense or sparse"
    )
    parser.add_argument(
        "--last-layer-dense", action="store_true", help="Last layer dense or sparse"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        help="Label smoothing to use, default 0.0",
        default=None,
    )
    parser.add_argument(
        "--first-layer-type", type=str, default=None, help="Conv type of first layer"
    )
    parser.add_argument(
        "--trainer", type=str, default="default", help="cs, ss, anomaly, or standard training"
    )
    parser.add_argument(
        "--score-init-constant",
        type=float,
        default=None,
        help="Sample Baseline Subnet Init",
    )
    parser.add_argument(
        "--prune_rate_epoch", 
        default=10, 
        type=int, 
        help="When pruning globally, scale up prune rate over this number of epochs"
    )
    parser.add_argument("--gaussian_aug", action="store_true", default=False, help="Gaussian noise augmentation to be added to the images")
    parser.add_argument("--std_gauss", help="Variance of sampled Gaussian noise for augmentation scheme", default=0.1, type=float)
    parser.add_argument("--p_clean", help="Fraction of clean data in gaussian augmentation scheme", default=1.0, type=float)
    parser.add_argument("--augmix", action="store_true", default=False, help="Use Augmix during training")
    parser.add_argument(
        "--jsd", action="store_true", default=False, help="Use Jensen-Shannon Divergence in loss with Augmix/Gaussian augmentation schemes"
    )
    parser.add_argument(
        "--all-augmix-augmentations", action="store_true", default=False, help="Use all Augmix augmentations when true (not recommended)"
    )
    parser.add_argument(
        '--mixture-width',
        default=3,
        type=int,
        help='Number of augmentation chains to mix per augmented example')
    parser.add_argument(
        '--mixture-depth',
        default=-1,
        type=int,
        help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
    parser.add_argument(
        '--aug-severity',
        default=3,
        type=int,
        help='Severity of base augmentation operators')

    # Updated for use with hpbandster
    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    # strip preceding hyphens from unknown arguments
    for i in range(len(unknown)):
      unknown[i] = _parser.arg_to_varname(unknown[i])

    # Allow for use from notebook without config file
    if len(sys.argv) > 1:
        get_config(args, unknown)

    # If pretrained model provided, check conv_type to set before building from config
    if args.pretrained:
      if os.path.isfile(args.pretrained):
        print("=> checking conv_type of pretrained model from '{}'".format(args.pretrained))
        pretrained_dict = torch.load(
            args.pretrained,
            map_location=torch.device("cuda:{}".format(args.multigpu[0])),
        )
        try:
          # Set conv_type argument to conv_type of pretrained model
          args.conv_type = pretrained_dict["conv_type"]
        except:
          print("=== WARNING: Pretrained model file does not contain 'conv_type' key. ===")
          print("=== WARNING: This may result in incorrect model being loaded. ===")
          print("=== SOLUTION: Either pass the conv_type used when training the model in arguments using the --conv-type flag or ensure that the correct conv_type is listed in the provided config file.")
        try:
          # Set prune_rate argument to prune_rate of pretrained model
          args.prune_rate = pretrained_dict["prune_rate"]
        except:
          print("=== WARNING: Pretrained model file does not contain 'prune_rate' key. ===")
          print("=== WARNING: This may result in incorrect model being loaded. ===")
          print("=== SOLUTION: Either pass the prune_rate used when training the model in arguments using the --prune-rate flag or ensure that the correct prune_rate is listed in the provided config file.")

    return args


def get_config(args, unknown):
    # get commands from command line
    override_args = _parser.argv_to_vars(sys.argv)

    # remove unknown args
    override_args = [a for a in override_args if a not in unknown]

    # load yaml file
    yaml_txt = open(args.config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.config}")
    args.__dict__.update(loaded_yaml)


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()
