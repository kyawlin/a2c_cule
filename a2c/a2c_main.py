import os
import sys

_path = os.path.abspath(os.path.pardir)
if not _path in sys.path:
    sys.path = [_path] + sys.path

from utils.launcher import main

def vtrace_parser_options(parser):

    parser.add_argument('--c-hat', type=int, default=1.0, help='Trace cutting truncation level (default: 1.0)')
    parser.add_argument('--rho-hat', type=int, default=1.0, help='Temporal difference truncation level (default: 1.0)')
    parser.add_argument('--num-minibatches', type=int, default=16, help='number of mini-batches in the set of environments (default: 16)')
    parser.add_argument('--num-steps-per-update', type=int, default=1, help='number of steps per update (default: 1)')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--lr-scale', action='store_true', default=False, help='Scale the learning rate with the batch-size')
    parser.add_argument('--num-stack', type=int, default=4, help='number of images in a stack (default: 4)')
    parser.add_argument('--num-steps', type=int, default=20, help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--tau', type=float, default=1.00, help='parameter for GAE (default: 1.00)')
    parser.add_argument('--use-gae', action='store_true', default=False, help='use generalized advantage estimation')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='value loss coefficient (default: 0.5)')

    parser.add_argument('--history-length', type=int, default=4, help='Number of consecutive states processed')
    parser.add_argument('--memory-capacity', type=int, default=int(12e3), metavar='CAPACITY', help='Experience replay memory capacity (default: 1,000,000)')
    #parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return (default: 3)')

    return parser

def vtrace_main():
    if sys.version_info.major == 3:
        from train import worker
    else:
        worker = None

    sys.exit(main(vtrace_parser_options, worker))

if __name__ == '__main__':
    vtrace_main()
