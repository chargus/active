"""
A library of functions used to run Langevin dynamics for the Vicsek model.

"""

import numpy as np
np.random.seed(0)


def initialize(n, rho):
    """Set up initial positions and angles in the simulation box.

    Parameters
    ----------
    n : int
        Number of particles in the simulation
    rho : float
        Density
    """
    L = np.sqrt(n / rho)
    pos = np.random.uniform(0, L, size=(n, 2))
    thetas = np.random.uniform(-np.pi, np.pi, size=n)
    return pos, thetas, L


def va(thetas):
    xs = np.cos(thetas)
    ys = np.sin(thetas)
    return np.sqrt(np.mean(xs) + np.mean(ys))


def va_traj(ttraj):
    xs = np.cos(ttraj)
    ys = np.sin(ttraj)
    return np.sqrt(np.mean(xs, axis=1)**2 + np.mean(ys, axis=1)**2)


def avg_theta(thetas):
    """Compute average orientation.

    Note: Cannot simply take the mean of all theta values, since this
    does not correclty account for the discontinuity from theta = -pi to pi.

    """
    return np.arctan2(np.mean(np.sin(thetas)), np.mean(np.cos(thetas)))


def apply_pbc(pos, L):
    """Apply periodic boundary conditions to an array of positions.

    It is assumed that the simulation box exists on domain [-L/2, L/2]^D
    where D is the number of dimensions.

    Parameters
    ----------
    pos : numpy ndarray
        Description
    L : float
        Description

    Returns
    -------
    pos:
        Description
    """
    return ((pos + L / 2) % L) - L / 2


def get_neighbors(pos, rcut, L=1.0):
    """Find neighbors within cutoff distance, including across PBCs.
    (returns a mask)
    """
    dx = np.subtract.outer(pos[:, 0], pos[:, 0])
    dy = np.subtract.outer(pos[:, 1], pos[:, 1])

    # Apply "minimum image" convention: interact with nearest periodic image
    dx = apply_pbc(dx, L)  # x component of i-j vector
    dy = apply_pbc(dy, L)  # y component of i-j vector

    r2 = dx**2 + dy**2  # Squared distance between all particle pairs

    # Select interaction pairs within cutoff distance
    # (also ignore self-interactions)
    mask = r2 < rcut**2  # Note: include self in alignment average.
    return mask


def align(pos, thetas, eta, rcut, L=1.0, mod=False):
    """Align all particles to the velocity vector of their neighbors.
    """
    mask = get_neighbors(pos, rcut, L)
    thetas_new = np.empty_like(thetas)
    # note: no easy way to vectorize for-loop, since len of thetas_new varies
    for i in range(len(thetas)):
        neighbors_thetas = thetas[mask[i]]
        avg = avg_theta(neighbors_thetas)
        thetas_new[i] = avg
    if mod:
        rho_global = len(mask) / L**2
        rho_local = sum(mask) / rcut**2
        eta = (eta * rho_local / rho_global)  # eta is different for each atom
    thetas_new += np.random.uniform(-eta / 2., eta / 2., len(thetas))
    return thetas_new


def timestep(pos, thetas, rcut, eta, vel, L, mod):
    """Update position and angles of all particles.
    """
    # First update the positions:
    dx = vel * np.cos(thetas)
    dy = vel * np.sin(thetas)
    pos[:, 0] = (pos[:, 0] + dx) % L  # Modulo L to handle PBCs
    pos[:, 1] = (pos[:, 1] + dy) % L  # Modulo L to handle PBCs

    # Now update the angles:
    thetas = align(pos, thetas, eta, rcut, L, mod)
    return pos, thetas


def run(n, rho, eta, vel, rcut=None, nframes=100, nlog=10, mod=False):
    """Run a dynamics simulation.

    Parameters
    ----------
    n : int
        Number of particles.
    rho : float
        Density (n / L**2).
    eta : float
        Specifies range for angular noise: dtheta ~ [-eta/2, eta/2]
    rcut : float
        Factor for cutoff radius beyond which interactions are not considered.
        This is a factor of L, e.g. 0.5 gives half of L as the the cutoff.
        If None, value of 0.5 is used
    nframes : int
        Number of frames for which to run simulation.
    nlog : int
        Log positions and kinetic energy to trajectories every `nlog` frames.
    mod : bool
        If true, use local density noise modification.
    """
    pos, thetas, L = initialize(n, rho)
    if rcut > L / 2:
        raise ValueError("rcut must be less than or equal to L/2 to satisfy"
                         "the minimum image convention.")
    if rcut is None:
        rcut = 0.5 * L
    rcut = rcut * L

    # Initialize arrays:
    ptraj = np.empty((nframes / nlog, pos.shape[0], pos.shape[1]))
    ttraj = np.empty((nframes / nlog, pos.shape[0]))

    # Begin iterating dynamics calculations:
    for i in range(nframes):
        if i % nlog == 0:
            ptraj[i / nlog] = pos
            ttraj[i / nlog] = thetas
        pos[:], thetas[:] = timestep(pos, thetas, rcut, eta, vel, L, mod)

    return ptraj, ttraj
