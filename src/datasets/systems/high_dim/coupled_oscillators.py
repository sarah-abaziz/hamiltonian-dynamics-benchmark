import numpy as np
from typing import Dict, Optional
from ...base import DynamicalSystem

class CoupledOscillators(DynamicalSystem):
    """N Coupled Harmonic Oscillators with nearest-neighbor coupling"""

    def __init__(self, n=3, k=1.0, k_coupling=0.1, m=1.0):
        super().__init__(f"{n} Coupled Oscillators", 2*n,
                        n=n, k=k, k_coupling=k_coupling, m=m)

    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        n = self.params['n']
        q, p = state[:n], state[n:]
        k, k_c, m = self.params['k'], self.params['k_coupling'], self.params['m']
        dqdt = p / m
        dpdt = -k * q  # Self springs
        # Coupling terms
        for i in range(n):
            if i > 0:
                dpdt[i] += k_c * (q[i-1] - q[i])
            if i < n-1:
                dpdt[i] += k_c * (q[i+1] - q[i])
        return np.concatenate([dqdt, dpdt])

    # Ideally we remove the python loops and handle batches of states.

    #def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
    #    self.validate_state(state)
    #    n = self.params['n']
    #    k, k_c, m = self.params['k'], self.params['k_coupling'], self.params['m']

    #    q, p = np.split(state, 2, axis=-1)

    #    # Kinetic part is already vectorized
    #    dqdt = p / m

    #    # Potential part (forces)
    #    dpdt = -k * q  # Force from individual springs

    #    # To vectorize coupling, create arrays of left and right neighbors
    #    q_left = np.pad(q, ((0, 0), (1, 0)), 'constant')[:, :-1]
    #    q_right = np.pad(q, ((0, 0), (0, 1)), 'constant')[:, 1:]

        # Compute forces from neighbors (handling boundaries via padding)
    #    force_left = k_c * (q_left - q)
    #    force_right = k_c * (q_right - q)

    #    # Set forces to 0 at the boundaries where there is no neighbor
    #    force_left[:, 0] = 0
    #    force_right[:, -1] = 0

    #    dpdt += force_left + force_right

    #    return np.hstack([dqdt, dpdt]).squeeze()


    def hamiltonian(self, state: np.ndarray) -> float:
        n = self.params['n']
        q, p = state[:n], state[n:]
        k, k_c, m = self.params['k'], self.params['k_coupling'], self.params['m']
        kinetic = np.sum(p**2) / (2 * m)
        potential = 0.5 * k * np.sum(q**2)  # Self springs
        # Coupling energy
        for i in range(n-1):
            potential += 0.5 * k_c * (q[i+1] - q[i])**2

        return kinetic + potential

    #def hamiltonian(self, state: np.ndarray) -> float:
    #    self.validate_state(state)
    #    n = self.params['n']
    #    k, k_c, m = self.params['k'], self.params['k_coupling'], self.params['m']

    #    q, p = np.split(state, 2, axis=-1)

    #    kinetic = np.sum(p**2, axis=-1) / (2 * m)
    #    potential_self = 0.5 * k * np.sum(q**2, axis=-1)

    #    # Vectorized coupling energy using np.diff
    #    # np.diff(q) calculates the difference between adjacent elements (q[i+1] - q[i]) for the entire array at once (no need for the for loop)
    #    potential_coupling = 0.5 * k_c * np.sum(np.diff(q, axis=-1)**2, axis=-1)

    #    return (kinetic + potential_self + potential_coupling).squeeze()

    def generate_initial_conditions(self, n_trajectories: int, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n, m = self.params['n'], self.params['m']
        q0 = rng.uniform(-0.3, 0.3, (n_trajectories, n))
        p0 = rng.uniform(-0.3*m, 0.3*m, (n_trajectories, n))
        return np.column_stack([q0, p0])

    def get_plot_config(self) -> Dict:
        n = self.params['n']
        return {
            'state_labels': [f'$q_{i+1}$' for i in range(n)] + [f'$p_{i+1}$' for i in range(n)],
            'phase_space': False,
            'x_range': [-1, 1],
            'y_range': [-1, 1]
        }
