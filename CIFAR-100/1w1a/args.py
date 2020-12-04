import argparse
import sys
import yaml

import parser as _parser

args = None
import networks

def parse_arguments():
    
    model_names = sorted(name for name in networks.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(networks.__dict__[name]))
    
    parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet34_1w1a',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                        ' (default: resnet34)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=350, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 20)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='', type=str)
    parser.add_argument('--save-every', dest='save_every',
                        help='Saves checkpoints at every specified number of epochs',
                        type=int, default=50)

    parser.add_argument(
        "--prune_rate",
        default=0.8,
        help="Amount of pruning to do during sparse training",
        type=float,
    )

    parser.add_argument(
        "--masksigntype", default="signmaskonlylayer", help="Policy for the learning rate."
    )    

    parser.add_argument("--mode", default="fan_in", help="Weight initialization mode")
    parser.add_argument(
        "--nonlinearity", default="relu", help="Nonlinearity used by initialization"
    )

    parser.add_argument(
        "--seed", default=1, help="seed number"
    )

    args = parser.parse_args()

    # Allow for use from notebook without config file
    if len(sys.argv) > 1:
        get_config(args)

    return args


def get_config(args):
    # get commands from command line
    override_args = _parser.argv_to_vars(sys.argv)

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
