"""
A python library of functions used to run molecular dynamics (MD) simulations.

The difficult computations are done using Numpy for performance reasons. The
name of this module (monad) is a reminder of its limitations -- this is
an instructional library used for monatomic particle simulations with no
long-range interactions.

Reduced units are used throughout, where the width and depth of the Lennard-
Jones potential are taken to be the characteristic length and energy,
respectively. The timestep dt and Boltzmann constant are set to unity. Where
the code implementation of a formula is obscured by "missing" variables
whose value are set to unity, an attempt has been made to note this in the
in-line comments.

"""

import numpy as np


def initialize_positions(n, L):
    """Set up initial positions in the simulation box.

    Try to place them somewhat uniformly in the box, without overlap.

    Parameters
    ----------
    n : int
        Number of particles in the simulation
    L : float
        Length of one side of the simulation box.
    """
    pos = np.zeros((n, 2), dtype=np.float32)
    nside = int(np.sqrt(n))  # For now, just assume sqrt(n) is integer
    lside = (4. / 5.) * L  # Shrink lattice to avoid overlap of periodic images
    x = np.linspace(-lside / 2, lside / 2, nside)
    y = np.linspace(-lside / 2, lside / 2, nside)
    X, Y = np.meshgrid(x, y)
    pos_init = np.array([X.flatten(), Y.flatten()]).T
    pos += pos_init
    pos += np.random.uniform(0, .03 * L, pos.shape)  # Rattle to break symmetry
    return pos


def initialize_velocities(n, width=0.2):
    """Initialize velocities to uniform distribution with zero mean

    Parameters
    ----------
    n : int
        Number of particles in the simulation
    width : float
        Width about zero with which to select uniformly random initial
        velocities.

    """
    velocity = np.zeros((n, 2), dtype=np.float32)
    velocity += np.random.uniform(-.2, .2, size=(n, 2))
    velocity -= np.mean(velocity, axis=0)  # Subtract the mean (no bulk flow)
    return velocity


def lj_potential(r2, sigma, epsilon):
    """Compute the Lennard-Jones potential energy for a pairwise interactions.

    Parameters
    ----------
    r2 : float or [N] numpy array
        Squared particle-particle separation.
    sigma : float
        Width of the potential, defined by distance at which potential is zero.
    epsilon : float
        Depth of the potential well, relative to the energy at infinite
        separation.

    """
    energy = 4 * epsilon * ((sigma**2 / r2)**6 - (sigma**2 / r2)**3)
    return energy


def lj_force(r2, sigma, epsilon):
    """Compute the Lennard-Jones force.

    Returned value is normalized by r, such that it can be multiplied by the
    x, y or z component of r to obtain the correct cartesian component of the
    force.

    Parameters
    ----------
    r2 : float or [N] numpy array
        Squared particle-particle separation.
    sigma : float
        Width of the potential, defined by distance at which potential is zero.
    epsilon : float
        Depth of the potential well, relative to the energy at infinite
        separation.

    """
    force = (48 / r2) * epsilon * \
        ((sigma**2 / r2)**6 - .5 * (sigma**2 / r2)**3)
    return force


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


def get_velocity(pos_next, pos_prev, L, dt):
    dpos = apply_pbc(pos_next - pos_prev, L)
    return dpos / (2. * dt)


def get_kinetic_energy_from_pos(pos_next, pos_prev, L, dt):
    return np.mean(get_velocity(pos_next, pos_prev, L, dt)**2, axis=0)


def get_kinetic_energy(vel):
    return np.mean(vel**2, axis=0)


def verlet_timestep(pos, pos_prev, dt, net_fx, net_fy, L,
                    sigma, epsilon):
    pos_next = np.empty_like(pos)
    dpos = pos - pos_prev
    pos_next[:, 0] = apply_pbc(pos[:, 0] + dpos[:, 0] + net_fx * dt**2, L)
    pos_next[:, 1] = apply_pbc(pos[:, 1] + dpos[:, 1] + net_fy * dt**2, L)
    pos_prev[:] = pos
    pos[:] = pos_next


def velocity_verlet_timestep(pos, vel, dt, forces, L, sigma, epsilon, rcut):
    """Advance positions and velocities one timestep using Velocity Verlet

    Velocity Verlet is similar to the basic Verlet algorithm, but explicitly
    tracks velocities as dynamics are advanced. Velocities (i.e. momenta) and
    positions are updated together in a "half-kick, drift, half-kick" scheme
    derived from Trotter factorization of the Liouville operator.

    """
    vel = vel + 0.5 * dt * forces
    pos = apply_pbc(pos + dt * vel, L)
    forces = get_forces(pos, L, sigma, epsilon, rcut)
    vel = vel + 0.5 * dt * forces

    return pos, vel


def get_forces(pos, L, sigma, epsilon, rcut):
    """Compute Lennard-Jones force on each particle.


    Parameters
    ----------
    pos : 2D numpy array [N x 3]
        Numpy array of particle positions.

    Returns
    -------
    forces: 2D numpy array [N x 3]
        Numpy array of net force vector acting on each particle.

    """
    # Determine particles within cutoff radius
    dx = np.subtract.outer(pos[:, 0], pos[:, 0])
    dy = np.subtract.outer(pos[:, 1], pos[:, 1])

    # Apply "minimum image" convention: interact with nearest periodic image
    dx = apply_pbc(dx, L)
    dy = apply_pbc(dy, L)

    r2 = dx**2 + dy**2  # Squared distance between all pairs of particles

    # Select interaction pairs within cutoff distance
    # (also ignore self-interactions)
    mask = r2 < rcut**2
    mask *= r2 > 0

    # Compute forces
    fx = np.zeros_like(dx)
    fx[mask] = dx[mask] * lj_force(r2[mask], sigma, epsilon)
    fy = np.zeros_like(dy)
    fy[mask] = dy[mask] * lj_force(r2[mask], sigma, epsilon)
    net_fx = np.sum(fx, axis=0)
    net_fy = np.sum(fy, axis=0)
    forces = np.stack([net_fx, net_fy], axis=1)
    return forces


def run(init_pos, init_vel, L, nframes, dt=0.005, nlog=10, rcut=None,
        sigma=1., epsilon=1., verbose=False):
    """Run a dynamics simulation.

    Parameters
    ----------
    init_pos : 2D numpy array [N x 3]
        Array of initial positions from which to start simulation.
    init_vel : 2D numpy array [N x 3]
        Array of initial velocities with which to start simulation.
    L : float
        Length of one side of the simulation box.
    nframes : int
        Number of frames to run simulation for
    dt : float
        Time step between frames in reduced units.
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
    vel = init_vel
    ptraj = np.empty((nframes / nlog, pos.shape[0], pos.shape[1]))
    vtraj = np.empty((nframes / nlog, pos.shape[0], pos.shape[1]))
    # vetraj = np.empty((nframes / nlog))
    # ketraj = np.empty((nframes / nlog, pos.shape[1]))

    # Begin iterating dynamics calculations:
    for i in range(nframes):
        forces = get_forces(pos, L, sigma, epsilon, rcut)
        pos[:], vel[:] = velocity_verlet_timestep(pos, vel, dt, forces, L,
                                                  sigma, epsilon, rcut)
        if i % nlog == 0:
            ptraj[i / nlog] = pos
            vtraj[i / nlog] = vel
            # if verbose:
            #     print pos[0, 0], pos_prev[0, 0], pos[0, 0] - pos_prev[0, 0]

    return ptraj, vtraj
