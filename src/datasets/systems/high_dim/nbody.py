import numpy as np
from typing import Optional, List
from dataclasses import dataclass
from ...base import DynamicalSystem


@dataclass
class CelestialBody:
    """Represents a celestial body with physical and visualization properties"""
    name: str
    mass: float
    initial_position: np.ndarray  # 3D position vector
    initial_momentum: np.ndarray  # 3D momentum vector
    color: str = 'blue'
    marker: str = 'o'
    markersize: float = 5


class NBodySystem(DynamicalSystem):
    """N-body gravitational system in Hamiltonian formulation"""

    def __init__(self, bodies: List[CelestialBody], use_barycenter_frame: bool = True):
        """
        Initialize N-body system.

        Args:
            bodies: List of CelestialBody objects
            use_barycenter_frame: If True, shift to barycenter rest frame
        """
        self.bodies = bodies
        self.n_bodies = len(bodies)

        # State dimension: 3 positions + 3 momenta per body
        state_dim = 6 * self.n_bodies

        # Store masses for efficient computation
        self.masses = np.array([body.mass for body in bodies])

        # Gravitational constant in AU³/(M_sun * day²)
        # For normalized units where solar mass = 1, distance in AU, time in days
        self.G = 2.959e-4

        # Process initial conditions
        initial_positions = np.array([body.initial_position for body in bodies])
        initial_momenta = np.array([body.initial_momentum for body in bodies])

        if use_barycenter_frame:
            # Shift to barycenter rest frame
            total_mass = np.sum(self.masses)
            barycenter_pos = np.sum(initial_positions * self.masses[:, np.newaxis], axis=0) / total_mass
            barycenter_vel = np.sum(initial_momenta, axis=0) / total_mass

            initial_positions -= barycenter_pos
            for i in range(self.n_bodies):
                initial_momenta[i] -= self.masses[i] * barycenter_vel

        # Create initial state vector [q1, q2, ..., qn, p1, p2, ..., pn]
        self.initial_state = np.concatenate([
            initial_positions.flatten(),
            initial_momenta.flatten()
        ])

        # Initialize parent class
        super().__init__(
            name=f"{self.n_bodies}-Body System",
            state_dim=state_dim
        )

    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Compute time derivatives using Hamilton's equations for N-body system.

        Args:
            t: Current time (unused for autonomous system)
            state: State vector [q1, ..., qn, p1, ..., pn]

        Returns:
            Time derivative [dq/dt, dp/dt]
        """
        # Split state into positions and momenta
        n = 3 * self.n_bodies
        q = state[:n].reshape(self.n_bodies, 3)
        p = state[n:].reshape(self.n_bodies, 3)

        # Compute dq/dt = ∂H/∂p = p/m
        dq_dt = np.zeros_like(q)
        for i in range(self.n_bodies):
            dq_dt[i] = p[i] / self.masses[i]

        # Compute dp/dt = -∂H/∂q = F (gravitational forces)
        dp_dt = np.zeros_like(p)
        for i in range(self.n_bodies):
            for j in range(i + 1, self.n_bodies):
                # Vector from i to j
                r_ij = q[j] - q[i]
                r_mag = np.linalg.norm(r_ij)

                if r_mag > 0:  # Avoid division by zero
                    # Gravitational force magnitude times direction
                    f_ij = self.G * self.masses[i] * self.masses[j] / (r_mag**3) * r_ij

                    # Newton's third law
                    dp_dt[i] += f_ij
                    dp_dt[j] -= f_ij

        # Combine derivatives
        return np.concatenate([dq_dt.flatten(), dp_dt.flatten()])

    def hamiltonian(self, state: np.ndarray) -> float:
        """
        Compute the Hamiltonian (total energy) of the N-body system.
        H = T + V where T is kinetic energy and V is potential energy.

        Args:
            state: State vector [q, p]

        Returns:
            Total energy H(q, p)
        """
        # Split state
        n = 3 * self.n_bodies
        q = state[:n].reshape(self.n_bodies, 3)
        p = state[n:].reshape(self.n_bodies, 3)

        # Kinetic energy T = Σ p²/(2m)
        kinetic = 0.0
        for i in range(self.n_bodies):
            kinetic += np.sum(p[i]**2) / (2 * self.masses[i])

        # Potential energy V = -Σ Gm₁m₂/r₁₂
        potential = 0.0
        for i in range(self.n_bodies):
            for j in range(i + 1, self.n_bodies):
                r_ij = np.linalg.norm(q[j] - q[i])
                if r_ij > 0:
                    potential -= self.G * self.masses[i] * self.masses[j] / r_ij

        return kinetic + potential

    def generate_initial_conditions(self, n_trajectories: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate initial conditions by perturbing the default initial state.

        Args:
            n_trajectories: Number of initial conditions to generate
            seed: Random seed for reproducibility

        Returns:
            Array of shape (n_trajectories, state_dim)
        """
        if seed is not None:
            np.random.seed(seed)

        # Start with the configured initial state
        initial_conditions = np.tile(self.initial_state, (n_trajectories, 1))

        # Add small random perturbations (1% of the initial values)
        perturbation_scale = 0.01
        perturbations = np.random.randn(n_trajectories, self.state_dim) * perturbation_scale

        # Scale perturbations by the magnitude of initial values
        # Avoid zero division by adding small epsilon
        scale = np.abs(self.initial_state) + 1e-10
        initial_conditions += perturbations * scale

        return initial_conditions