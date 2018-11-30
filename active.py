"""
A library of functions used to run Langevin dynamics for the Vicsek model.

"""

import numpy as np
import monad


def initialize(n, L=1.):
    """Set up initial positions and angles in the simulation box.

    Parameters
    ----------
    n : int
        Number of particles in the simulation
    L : float
        Length of one side of the simulation box.
    """
    pos = np.random.uniform(0, 1, size=(n, 2))
    thetas = np.random.uniform(-np.pi, np.pi, size=n)
    return pos, thetas


def avg_theta(thetas):
    """Compute average orientation.

    Note: Cannot simply take the mean of all theta values, since this
    does not correclty account for the discontinuity from theta = -pi to pi.

    """
    return np.arctan(np.mean(np.sin(thetas)) / np.mean(np.cos(thetas)))


def get_neighbors(pos, rcut, L=1.0):
    """Find neighbors within cutoff distance, including across PBCs.
    (returns a mask)
    """
    dx = np.subtract.outer(pos[:, 0], pos[:, 0])
    dy = np.subtract.outer(pos[:, 1], pos[:, 1])

    # Apply "minimum image" convention: interact with nearest periodic image
    dx = monad.apply_pbc(dx, L)  # x component of i-j vector
    dy = monad.apply_pbc(dy, L)  # y component of i-j vector

    r2 = dx**2 + dy**2  # Squared distance between all particle pairs

    # Select interaction pairs within cutoff distance
    # (also ignore self-interactions)
    mask = r2 < rcut**2  # Note: include self in alignment average?
    return mask


def align(pos, thetas, eta, rcut, L=1.0):
    """Align all particles to the velocity vector of their neighbors.
    """
    mask = get_neighbors(pos, rcut, L)
    thetas_new = np.empty_like(thetas)
    # note: no easy way to vectorize for-loop, since len of thetas_new varies
    for i in range(len(thetas)):
        neighbors_thetas = thetas[mask[i]]
        avg = avg_theta(neighbors_thetas)
        thetas_new[i] = avg
    thetas_new += np.random.uniform(-np.pi, np.pi, len(thetas)) * eta
    return thetas_new


def timestep(pos, thetas, rcut, eta, vel, L):
    """Update position and angles of all particles.
    """
    # First update the positions:
    dx = vel * np.cos(thetas)
    dy = vel * np.sin(thetas)
    pos[:, 0] = (pos[:, 0] + dx) % L  # Modulo L to handle PBCs
    pos[:, 1] = (pos[:, 1] + dy) % L  # Modulo L to handle PBCs

    # Now update the angles:
    thetas = align(pos, thetas, eta, rcut, L)
    return pos, thetas


def run(init_pos, init_thetas, eta, vel, L, nframes, nlog=10, rcut=None):
    """Run a dynamics simulation.

    Parameters
    ----------
    init_pos : 2D numpy array [N x 3]
        Array of initial positions from which to start simulation.
    init_thetas : 2D numpy array [N x 3]
        Array of initial velocities with which to start simulation.
    L : float
        Length of one side of the simulation box.
    nframes : int
        Number of frames to run simulation for
    nlog : int
        Log positions and kinetic energy to trajectories every `nlog` frames.
    rcut : float
        Cutoff radius beyond which interactions are not considered. Must be
        less than or equal to L/2.
    """
    if rcut > L / 2:
        raise ValueError("rcut must be less than or equal to L/2 to satisfy"
                         "the minimum image convention.")
    if rcut is None:
        rcut = L / 2.

    # Initialize arrays:
    pos = init_pos
    thetas = init_thetas
    ptraj = np.empty((nframes / nlog, pos.shape[0], pos.shape[1]))
    ttraj = np.empty((nframes / nlog, pos.shape[0]))

    # Begin iterating dynamics calculations:
    for i in range(nframes):
        if i % nlog == 0:
            ptraj[i / nlog] = pos
            ttraj[i / nlog] = thetas
        pos[:], thetas[:] = timestep(pos, thetas, rcut, eta, vel, L)

    return ptraj, ttraj
