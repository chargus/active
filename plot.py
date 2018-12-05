#!/usr/bin/env python


import numpy as np
import active
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

etas = np.linspace(0, 5, 21)
ns = [40, 100, 400, 1000]
for n in ns:
    vas = []
    for eta in etas:
        path = 'sim1/eta_{}n_{}'.format(eta, n)
        ttraj = np.load(path + '_ttraj.npy')
        ptraj = np.load(path + '_ptraj.npy')
        avg_order_param = np.mean(active.va_traj(ttraj)[200:])
        vas.append(avg_order_param)
    ax.plot(etas, vas, label='N={}'.format(n))
ax.legend()
ax.set_xlabel('$\eta$', fontsize=20)
ax.set_ylabel('$v_a$  ', fontsize=20, rotation=0)
fig.savefig('out.png', dpi=400, bbox_inches='tight')
