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
import nve_basic


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
    pos = np.zeros((n, 3), dtype=np.float64)
    nside = int(np.sqrt(n))  # For now, just assume sqrt(n) is integer
    lside = (4. / 5.) * L  # Shrink lattice to avoid overlap of periodic images
    x = np.linspace(-lside / 2, lside / 2, nside)
    y = np.linspace(-lside / 2, lside / 2, nside)
    z = np.linspace(-lside / 2, lside / 2, nside)

    X, Y, Z = np.meshgrid(x, y, z)
    pos_init = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
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
    velocity = np.zeros((n, 3), dtype=np.float64)
    velocity += np.random.uniform(-.2, .2, size=(n, 3))
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


def lj(r2, sigma, epsilon):
    """Compute the Lennard-Jones force and potential energy.

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
    r12 = (sigma**2 / r2)**6
    r6 = (sigma**2 / r2)**3
    lj_force = (48 / r2) * epsilon * (r12 - .5 * r6)
    lj_energy = 4 * epsilon * (r12 - r6)
    return lj_force, lj_energy


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


def verlet_timestep(pos, pos_prev, dt, net_fx, net_fy, net_fz, L,
                    sigma, epsilon):
    pos_next = np.empty_like(pos)
    dpos = pos - pos_prev
    pos_next[:, 0] = apply_pbc(pos[:, 0] + dpos[:, 0] + net_fx * dt**2, L)
    pos_next[:, 1] = apply_pbc(pos[:, 1] + dpos[:, 1] + net_fy * dt**2, L)
    pos_next[:, 2] = apply_pbc(pos[:, 2] + dpos[:, 2] + net_fz * dt**2, L)
    pos_prev[:] = pos
    pos[:] = pos_next


def velocity_verlet_timestep(pos, vel, dt, old_forces, L,
                             sigma, epsilon, rcut):
    """Advance positions and velocities one timestep using Velocity Verlet

    Velocity Verlet is similar to the basic Verlet algorithm, but explicitly
    tracks velocities as dynamics are advanced. Velocities (i.e. momenta) and
    positions are updated together in a "half-kick, drift, half-kick" scheme
    derived from Trotter factorization of the Liouville operator.

    """
    # old_forces, energy = get_forces(pos, L, sigma, epsilon, rcut)
    vel = vel + 0.5 * dt * old_forces
    pos = apply_pbc(pos + dt * vel, L)
    new_forces, energy = get_forces(pos, L, sigma, epsilon, rcut)
    vel = vel + 0.5 * dt * new_forces

    return pos, vel, new_forces, energy


def velocity_verlet_timestep2(pos, vel, dt, old_forces, L,
                              sigma, epsilon, rcut, get_t_and_p=False):
    """Second implementation of velocity verlet.
    """
    old_forces, energy = get_forces(pos, L, sigma, epsilon, rcut)
    pos = apply_pbc(pos + dt * vel + .5 * old_forces * dt**2, L)
    if get_t_and_p:
        new_forces, energy, T, p = get_forces(pos, L, sigma, epsilon,
                                              rcut, vel)
        res = (energy, T, p)
    else:
        new_forces, energy = get_forces(pos, L, sigma, epsilon, rcut)
        res = (energy,)

    vel = vel + 0.5 * dt * (old_forces + new_forces)
    return (pos, vel, new_forces) + res


def get_forces(pos, L, sigma, epsilon, rcut, vel=None):
    """Compute Lennard-Jones force on each particle.


    Parameters
    ----------
    pos : 2D numpy array [N x 3]
        Numpy array of particle positions.
    vel : 2D numpy array [N x 3], optional
        Numpy array of velocities. If provided, returned values include
        temperature and pressure.

    Returns
    -------
    forces: 2D numpy array [N x 3]
        Numpy array of net force vector acting on each particle.
    energy: float
        Potential energy from LJ interaction.
    temperature: float, optional
        Instantaneous temperature of the system.
    pressure: float, optional
        Instantaneous pressure of the system.

    """
    # Determine particles within cutoff radius
    dx = np.subtract.outer(pos[:, 0], pos[:, 0])
    dy = np.subtract.outer(pos[:, 1], pos[:, 1])
    dz = np.subtract.outer(pos[:, 2], pos[:, 2])

    # Apply "minimum image" convention: interact with nearest periodic image
    dx = apply_pbc(dx, L)  # x component of i-j vector
    dy = apply_pbc(dy, L)  # y component of i-j vector
    dz = apply_pbc(dz, L)  # y component of i-j vector

    r2 = dx**2 + dy**2 + dz**2  # Squared distance between all particle pairs

    # Select interaction pairs within cutoff distance
    # (also ignore self-interactions)
    mask = r2 < rcut**2
    mask *= r2 > 0

    # Compute forces
    lj_force, lj_energy = lj(r2[mask], sigma, epsilon)
    fx = np.zeros_like(dx)
    fx[mask] = -dx[mask] * lj_force  # Negative sign so dx points from j to i
    fy = np.zeros_like(dy)
    fy[mask] = -dy[mask] * lj_force  # Negative sign so dy points from j to i
    fz = np.zeros_like(dz)
    fz[mask] = -dz[mask] * lj_force  # Negative sign so dz points from j to i
    net_fx = np.sum(fx, axis=0)
    net_fy = np.sum(fy, axis=0)
    net_fz = np.sum(fz, axis=0)
    forces = np.stack([net_fx, net_fy, net_fz], axis=1)
    energy = np.sum(lj_energy)
    if vel is not None:
        temperature = np.sqrt(np.sum(vel**2) / (3 * len(vel)))
        rho = len(pos) / L**3
        # px = np.dot(fx.ravel(), dx.ravel())
        # py = np.dot(fy.ravel(), dy.ravel())
        # pz = np.dot(fz.ravel(), dz.ravel())
        # pressure = rho * temperature + (1. / 6.) * (px + py + pz)
        virial = np.dot(lj_force, r2[mask].ravel())
        pressure = rho * temperature + virial / (6. * L**3)
        return forces, energy, temperature, pressure
    else:
        return forces, energy


def run(n, rho, T0, nframes, nframes_eq=50, dt=5e-3, nlog=10, rcut=None,
        sigma=1., epsilon=1., verbose=False):
    """Run a dynamics simulation.

    Parameters
    ----------
    n : int
        Number of particles
    rho : float
        Density
    T0 : float
        Initial temperature. May fluctuate and drift.
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

    pos, L = nve_basic.fcc_positions(n, rho)
    vel = initialize_velocities(n)
    if rcut > L / 2:
        raise ValueError("rcut must be less than or equal to L/2 to satisfy"
                         "the minimum image convention.")
    if rcut is None:
        rcut = L / 2.

    # Equilibrate (don't keep any of this data):
    for i in range(nframes_eq):
        forces, ene = get_forces(pos, L, sigma, epsilon, rcut)
        pos[:], vel[:], forces[:], ene = velocity_verlet_timestep2(
            pos, vel, dt, forces, L, sigma, epsilon, rcut)

    # Now set the temperature:
    temp_factor = np.sqrt(np.sum(vel**2) / (3 * len(vel) * T0))
    vel = vel / temp_factor  # Scale velocities to match desired temp

    # Initialize arrays:
    forces, ene, T, p = get_forces(pos, L, sigma, epsilon, rcut, vel)
    ptraj = np.empty((nframes / nlog, pos.shape[0], pos.shape[1]))
    vtraj = np.empty((nframes / nlog, pos.shape[0], pos.shape[1]))
    etraj = np.empty(nframes / nlog)
    temptraj = np.empty(nframes / nlog)
    prestraj = np.empty(nframes / nlog)

    # Begin iterating dynamics calculations:
    for i in range(nframes):
        if i % nlog == 0:
            ptraj[i / nlog] = pos
            vtraj[i / nlog] = vel
            etraj[i / nlog] = ene
            temptraj[i / nlog] = T
            prestraj[i / nlog] = p
            pos[:], vel[:], forces[:], ene, T, p = velocity_verlet_timestep2(
                pos, vel, dt, forces, L, sigma, epsilon, rcut, True)
        else:
            pos[:], vel[:], forces[:], ene = velocity_verlet_timestep2(
                pos, vel, dt, forces, L, sigma, epsilon, rcut)

    return ptraj, vtraj, etraj, temptraj, prestraj
