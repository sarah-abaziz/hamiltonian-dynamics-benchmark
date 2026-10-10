import numpy as np
from typing import Dict, Optional, Tuple
from ...base import DynamicalSystem


class HenonHeiles(DynamicalSystem):
    """Hénon–Heiles system (2 degrees of freedom), state [q1, q2, p1, p2].

        H = (p1² + p2²)/2 + (q1² + q2²)/2 + lam * (q1² q2 - q2³/3)

    A classic non-integrable benchmark: for small energy the motion is regular, and it
    becomes increasingly chaotic as E grows. Orbits stay bounded only for
    E < E_esc = 1 / (6 lam²) (= 1/6 for lam = 1); above that they can escape to infinity.
    """

    def __init__(self, lam: float = 1.0, energy_range: Tuple[float, float] = (0.02, 0.15)):
        e_min, e_max = energy_range
        if not 0 < e_min < e_max:
            raise ValueError("energy_range must satisfy 0 < e_min < e_max")
        if e_max >= 1.0 / (6.0 * lam ** 2):
            raise ValueError(
                f"energy_range max {e_max} must be below the escape energy "
                f"1/(6 lam^2) = {1.0 / (6.0 * lam ** 2):.4f}, otherwise orbits can escape.")
        super().__init__("Henon-Heiles", 4, lam=lam, energy_range=tuple(energy_range))

    @property
    def escape_energy(self) -> float:
        return 1.0 / (6.0 * self.params["lam"] ** 2)

    def _potential(self, q1, q2):
        lam = self.params["lam"]
        return 0.5 * (q1 ** 2 + q2 ** 2) + lam * (q1 ** 2 * q2 - q2 ** 3 / 3.0)

    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        q1, q2, p1, p2 = state
        lam = self.params["lam"]
        dp1 = -q1 - 2.0 * lam * q1 * q2
        dp2 = -q2 - lam * (q1 ** 2 - q2 ** 2)
        return np.array([p1, p2, dp1, dp2])

    def hamiltonian(self, state: np.ndarray) -> float:
        q1, q2, p1, p2 = state
        return 0.5 * (p1 ** 2 + p2 ** 2) + self._potential(q1, q2)

    def generate_initial_conditions(self, n_trajectories: int, seed: Optional[int] = None) -> np.ndarray:
        """Sample states on prescribed energy levels E ~ U(energy_range).

        q1, q2 and p1 are drawn from a small box, E is drawn from energy_range, and p2 is
        solved from H = E:   p2² = 2(E - V(q)) - p1²   (sign of p2 chosen at random).
        Draws with p2² < 0 are rejected. The box lies inside the bounded basin of the
        potential, so every orbit stays bounded.
        """
        rng = np.random.default_rng(seed)
        e_min, e_max = self.params["energy_range"]
        s = 1.0 / self.params["lam"]                       # keeps the geometry for lam != 1

        out = []
        for _ in range(1000):                              # batches, to bound the loop
            m = max(4 * n_trajectories, 16)
            q1 = rng.uniform(-0.3 * s, 0.3 * s, m)
            q2 = rng.uniform(-0.3 * s, 0.3 * s, m)
            p1 = rng.uniform(-0.2 * s, 0.2 * s, m)
            E = rng.uniform(e_min, e_max, m)
            p2_sq = 2.0 * (E - self._potential(q1, q2)) - p1 ** 2
            ok = p2_sq >= 0
            p2 = np.sqrt(p2_sq[ok]) * rng.choice([-1.0, 1.0], ok.sum())
            out.append(np.column_stack([q1[ok], q2[ok], p1[ok], p2]))
            if sum(len(o) for o in out) >= n_trajectories:
                return np.concatenate(out)[:n_trajectories]
        raise RuntimeError("Could not sample enough initial conditions for this energy_range.")

    def get_plot_config(self) -> Dict:
        return {
            "state_labels": [r"$q_1$", r"$q_2$", r"$p_1$", r"$p_2$"],
            "phase_space": False,
            "x_range": [-0.6, 0.6],
            "y_range": [-0.6, 0.6],
        }
