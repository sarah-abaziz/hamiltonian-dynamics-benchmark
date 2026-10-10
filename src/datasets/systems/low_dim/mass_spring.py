import numpy as np
from typing import Optional
from ...base import DynamicalSystem

class MassSpringSystem(DynamicalSystem):
    """Mass-Spring System in Hamiltonian form."""

    def __init__(self, k=1.0, m=1.0, n_masses=1):
        super().__init__(f"{n_masses}-Mass Spring System",2, k=k, m=m)

    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        q, p = state
        k, m = self.params['k'], self.params['m']
        dqdt = p / m
        dpdt = -k * q
        return np.array([dqdt, dpdt])

    #def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        #self.validate_state(state)
        #q, p = np.split(state, 2, axis=-1)
        #k, m = self.params['k'], self.params['m']
        #dqdt = p / m
        #dpdt = -k * q
        #return np.hstack([dqdt, dpdt]).squeeze()

    def hamiltonian(self, state: np.ndarray) -> float:
        q, p = state
        k, m = self.params['k'], self.params['m']
        kinetic = (p**2) / (2 * m)
        potential = k * (q ** 2) / 2
        return kinetic + potential

    #def hamiltonian(self, state: np.ndarray) -> float:
        #self.validate_state(state)
        #q, p = np.split(state, 2, axis=-1)
        #k, m = self.params['k'], self.params['m']
        #kinetic = (p**2) / (2 * m)
        #potential = k * (q**2) / 2
        #return (kinetic + potential).squeeze()

    def generate_initial_conditions(self, n_trajectories: int, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        m = self.params['m']
        q0 = rng.uniform(-0.5, 0.5, (n_trajectories, 1))
        p0 = rng.uniform(-0.5*m, 0.5*m, (n_trajectories, 1))
        return np.hstack([q0, p0])

    def get_plot_config(self) -> dict:
      return {
        'state_labels': [r"$q$ (displacement)", r"$p$ (momentum)$"],
        'phase_space': True,
        'x_range': [-2, 2],
        'y_range': [-2, 2]
       }