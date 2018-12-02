#!/usr/bin/env python

"""Submission script for running Vicsek model calculations on cluster.
"""
import active
import numpy as np
np.random.seed(0)


def main(args):
    # Simulation settings
    n = args.n
    eta = args.eta
    nframes = args.nframes
    # Some hardcoded values for _all_ simulations:
    rho = 4.0
    vel = 0.1
    rcut = 1.
    nlog = 10
    ptraj, ttraj = active.run(n, rho, eta, vel, rcut, nframes, nlog)
    np.save(args.path + '_ptraj', ptraj)
    np.save(args.path + '_ttraj', ttraj)


def opts():
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', action='store', type=str,
                        help='Output file path.')
    parser.add_argument('-n', action='store', type=int, default=40,
                        help='Number of particles.')
    parser.add_argument('--nframes', action='store', type=int, default=1000,
                        help='Number of frames to simulate.')
    parser.add_argument('--eta', action='store', type=float, default=.2,
                        help='Value for eta (pseudo-temperature)')
    return parser


if __name__ == "__main__":
    args = opts().parse_args()
    main(opts().parse_args())
