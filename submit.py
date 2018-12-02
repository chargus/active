#!/usr/bin/env python

"""Batch queueing script.
"""

import subprocess
import shlex
import numpy as np


etas = np.linspace(0, 5, 21)
ns = [40, 100, 400, 1000]
nframes = [2000, 500, 300, 300]
for n in ns:
    for eta in etas:
        path = './simdir/' + 'eta_{}n_{}'.format(eta, n)
        cmd = 'qsub -q all.q run.py {} -n {} --nframes {} --eta {}'
        cmd = cmd.format(path, n, nframes, eta)
        # out = subprocess.check_output(shlex.split(cmd))
        print cmd
        # out
