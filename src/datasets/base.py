import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from .integrators import get_integrator

class DynamicalSystem(ABC):
    """Abstract base class for dynamical systems in Hamiltonian form"""
    
    default_integrator: str = "midpoint"

    def __init__(self, name: str, state_dim: int, **params):
        self.name = name
        self.state_dim = state_dim
        self.params = params

    @abstractmethod
    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """ 
        Compute the time derivative of the state using Hamilton's equations.
        State is [q1, ..., qn, p1, ..., pn] where q are configurations and p are momenta.

        Args:
            t: Current time
            state: Current state vector of dimension state_dim

        Returns:
            Time derivative of state vector [dq/dt, dp/dt]
        """
        pass

    @abstractmethod
    def hamiltonian(self, state: np.ndarray) -> float:
        """
        Compute the Hamiltonian (total energy) of the system.

        Args:
            state: Current state vector [q, p]

        Returns:
            Hamiltonian value H(q, p)
        """
        pass


    @abstractmethod
    def generate_initial_conditions(self, n_trajectories: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate random initial conditions in phase space.

        Args:
            n_trajectories: Number of initial conditions to generate

        Returns:
            Array of shape (n_trajectories, state_dim) with [q, p] format
        """
        pass

    def simulate_trajectory(self, initial_state: np.ndarray, tspan: tuple, steps: int, integrator: Optional[str] = None, **integrator_kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate a single trajectory of the system using the midpoint method.

        Args:
            initial_state (np.ndarray): Initial state vector of shape (state_dim,).
            tspan (tuple): Time span for simulation (t0, tf).
            steps (int): Number of integration steps.

        Returns:
            tuple:
                - t (np.ndarray): Array of time points of shape (steps+1,).
                - y (np.ndarray): Array of states at each time point of shape (steps+1, state_dim).
        """
        # t, trajectory = midpoint(self.dynamics, tspan, initial_state, steps)
        # return t, trajectory
        fn = get_integrator(integrator or self.default_integrator)
        return fn(self.dynamics, tspan, initial_state, steps, **integrator_kwargs)

    def trajectory_energy(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Compute the Hamiltonian (energy) along a given trajectory.

        Args:
            trajectory (np.ndarray): Array of states along the trajectory, shape (n_steps, state_dim).

        Returns:
            np.ndarray: Array of energy values at each step, shape (n_steps,).
        """
        return np.array([self.hamiltonian(s) for s in trajectory])
        #return self.hamiltonian(trajectory) # This implementation is vectorized for efficiency. For very long trajectories, this Python-level loop can be a performance bottleneck. A more efficient, "NumPy-native" approach e.g. vectorization pushes the loop down into NumPy's optimized C code.

    def validate_state(self, state: np.ndarray) -> np.ndarray:
        """
        Validate the input state vector and ensure it has the correct dimensionality.

        Args:
            state (np.ndarray): Input state vector of shape (..., state_dim).

        Returns:
            np.ndarray: Validated state vector, unchanged if valid.

        Raises:
            ValueError: If the last dimension of `state` does not match `state_dim`.
        """
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"State dimension mismatch: expected {self.state_dim}, got {state.shape[-1]}")
        return state
