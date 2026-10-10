import numpy as np
from typing import Dict, Optional
from ...base import DynamicalSystem

class SimplePendulum(DynamicalSystem):
    """Simple Pendulum in Hamiltonian form: H = p²/(2mL²) + mgL(1-cos(q))"""

    def __init__(self, g: float = 9.81, L: float = 1.0, m: float = 1.0):
        super().__init__("Simple Pendulum", 2, g=g, L=L, m=m)

    #def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
    #    q, p = state
    #    m, g, L = self.params["m"], self.params["g"], self.params["L"]
    #    dqdt = p / (m * L**2)
    #    dpdt = -m * g * L * np.sin(q)
    #    return np.array([dqdt, dpdt])

    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        '''
        As it is currently written, the dynamics method is designed to work on a single state vector at a time.
        It will raise an error if we pass it a batch of states (e.g., an array of shape (100, 2)).
        '''
        self.validate_state(state) # Good practice
        q, p = np.split(state, 2, axis=-1) #  We can use np.split(state, 2, axis=-1) to unpack q and p. This works whether state is (2,) or (N, 2)
        m, g, L = self.params["m"], self.params["g"], self.params["L"]
        dqdt = p / (m * L**2)
        dpdt = -m * g * L * np.sin(q)
        return np.hstack([dqdt, dpdt]).squeeze() # np.hstack correctly reassembles the results

    #def hamiltonian(self, state: np.ndarray) -> float:
    #    q, p = state
    #    m, g, L = self.params["m"], self.params["g"], self.params["L"]
    #    kinetic = p**2 / (2 * m * L**2)
    #    potential = m * g * L * (1 - np.cos(q))
    #    return kinetic + potential

    def hamiltonian(self, state: np.ndarray) -> float:
        self.validate_state(state)
        q, p = np.split(state, 2, axis=-1)
        m, g, L = self.params["m"], self.params["g"], self.params["L"]
        kinetic = p**2 / (2 * m * L**2)
        potential = m * g * L * (1 - np.cos(q))
        return (kinetic + potential).squeeze()

    def generate_initial_conditions(self, n_trajectories: int, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        m, L = self.params["m"], self.params["L"]
        q0 = rng.uniform(-np.pi, np.pi, n_trajectories)
        # [-π/3, π/3] creates a dataset of purely oscillatory motions.
        # [-π, π] includes initial conditions near the top of the swing. This can lead to rotational motion (the pendulum swinging all the way around) and includes the unstable equilibrium point (starting straight up).
        p0 = rng.uniform(-2 * m * L**2, 2 * m * L**2, n_trajectories)
        return np.column_stack([q0, p0])

    def get_plot_config(self) -> Dict:
        m, L = self.params["m"], self.params["L"]
        return {
            "state_labels": [r"$q$ (angle)", r"$p$ (angular momentum)$"],
            "phase_space": True,
            "x_range": [-np.pi/3, np.pi/3],
            "y_range": [-2 * m * L**2, 2 * m * L**2],
        }


class DoublePendulum(DynamicalSystem):
    """Double Pendulum in Hamiltonian form"""

    def __init__(self, g=9.81, L1=1.0, L2=1.0, m1=1.0, m2=1.0):
        super().__init__("Double Pendulum", 4, g=g, L1=L1, L2=L2, m1=m1, m2=m2)

    def _mass_matrix(self, q1, q2):
        """Mass matrix and its determinant"""
        L1, L2 = self.params['L1'], self.params['L2']
        m1, m2 = self.params['m1'], self.params['m2']
        delta = q1 - q2
        cos_delta = np.cos(delta)
        M11 = (m1 + m2) * L1**2
        M12 = m2 * L1 * L2 * cos_delta
        M22 = m2 * L2**2
        detM = M11 * M22 - M12**2
        return M11, M12, M22, detM

    def _mass_matrix_inv(self, q1, q2):
        M11, M12, M22, detM = self._mass_matrix(q1, q2)
        Minv = (1.0 / detM) * np.array([[M22, -M12],
                                        [-M12, M11]])
        return Minv

    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        q1, q2, p1, p2 = state
        g = self.params['g']
        L1, L2 = self.params['L1'], self.params['L2']
        m1, m2 = self.params['m1'], self.params['m2']
        delta = q1 - q2
        sin_delta = np.sin(delta)
        cos_delta = np.cos(delta)
        Minv = self._mass_matrix_inv(q1, q2)
        dq1dt, dq2dt = Minv @ np.array([p1, p2])
        h1 = (p1 * p2 * sin_delta) / (L1 * L2 * (m1 + m2 * sin_delta ** 2))
        h2 = m2 * L2 * (p1 ** 2) + (m1 + m2) * (L1 ** 2) * (p2 ** 2) - 2 * m2 * L1 * L2 * p1 * p2 * cos_delta
        h2 /= 2 * (L1 * L2 * (m1 + m2 * sin_delta ** 2)) ** 2
        dp1dt = - (m1 + m2) * g * L1 *np.sin(q1) - h1 + h2 * np.sin(2* (q1 - q2))
        dp2dt = - m2 * g * L2 * np.sin(q2) + h1 - h2 * np.sin(2* (q1 - q2))
        return np.array([dq1dt, dq2dt, dp1dt, dp2dt])

    def hamiltonian(self, state: np.ndarray) -> float:
        q1, q2, p1, p2 = state
        g = self.params['g']
        L1, L2 = self.params['L1'], self.params['L2']
        m1, m2 = self.params['m1'], self.params['m2']
        Minv = self._mass_matrix_inv(q1, q2)
        p = np.array([p1, p2])
        kinetic = 0.5 * p @ Minv @ p
        potential = -(m1 + m2) * g * L1 * np.cos(q1) - m2 * g * L2 * np.cos(q2)
        return kinetic + potential

    def generate_initial_conditions(
            self,
            n_trajectories: int,
            seed: Optional[int] = None
        ) -> np.ndarray:
            """
            Generate random initial conditions for the double pendulum.
            Returns (q1, q2, p1, p2) where p = M(q) * dq.
            """
            rng = np.random.default_rng(seed)

            # Random positions and velocities
            q1_0 = rng.uniform(-np.pi / 3, np.pi / 3, n_trajectories)
            q2_0 = rng.uniform(-np.pi / 3, np.pi / 3, n_trajectories)
            dq1 = rng.uniform(-0.5, 0.5, n_trajectories)
            dq2 = rng.uniform(-0.5, 0.5, n_trajectories)

            dq = np.stack([dq1, dq2], axis=1)
            p = np.zeros_like(dq)

            # Use existing mass matrix function for each trajectory
            for i in range(n_trajectories):
                M11, M12, M22, det = self._mass_matrix(q1_0[i], q2_0[i])
                if abs(det) < 1e-12:
                    raise ValueError(f"Mass matrix nearly singular at sample {i}.")
                M = np.array([[M11, M12], [M12, M22]])
                p[i] = M @ dq[i]

            return np.column_stack([q1_0, q2_0, p[:, 0], p[:, 1]])


    def get_plot_config(self) -> Dict:
        return {
            "state_labels": [r"$q_1$", r"$q_2$", r"$p_1$", r"$p_2$"],
            "phase_space": False,
            "x_range": [-np.pi, np.pi],
            "y_range": [-5, 5],
        }