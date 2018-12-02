#!/usr/bin/env python

"""Batch queueing script.
"""

import subprocess
import shlex
import numpy as np

cmd = ('qsub -q all.q -pe orte 1 -S /bin/bash -N gp_mass_umb -pe mpi 1 -cwd '
       '-e log -o log -m be -M hargus@berkeley.edu '
       'run.py  {} -n {} --nframes {} --eta {}')
cmd = 'qsub -q all.q -cwd -V run.py {} -n {} --nframes {} --eta {}'

etas = np.linspace(0, 5, 21)
ns = [40, 100, 400, 1000]
nframes = [2000, 500, 300, 300]
for n, nframes in zip(ns, nframes):
    for eta in etas:
        path = 'sim/eta_{}n_{}'.format(eta, n)
        fcmd = cmd.format(path, n, nframes, eta)
        # out = subprocess.check_output(shlex.split(fcmd))
        print fcmd
        # out
