import numpy as np
from typing import Dict, Optional
from ...base import DynamicalSystem


class FermiPastaUlamTsingou(DynamicalSystem):
    """Fermi–Pasta–Ulam–Tsingou chain with m stiff + (m+1) soft springs (GNI, Section I.5.1).

    2m unit masses on a line, fixed walls at both ends, alternating soft (nonlinear) and
    stiff (linear) springs. State [q_1..q_2m, p_1..p_2m], dimension 4m (m=3 -> 12).

        H = 1/2 sum p_i^2  +  omega^2/4 * sum_{i=1..m} (q_2i - q_2i-1)^2          (stiff springs)
                           +  sum_{j=0..m} (q_2j+1 - q_2j)^4,   q_0 = q_2m+1 = 0   (soft, nonlinear)

    For large omega the stiff springs oscillate fast and exchange energy very slowly: the
    quantities I_j = 1/2 ydot_j^2 + omega^2/2 y_j^2 (see `stiff_energies`) are nearly conserved.
    """

    def __init__(self, m: int = 3, omega: float = 50.0, ic_mode: str = "canonical",
                 ic_noise: float = 0.05, box: float = 2.1):
        if m < 1:
            raise ValueError("m must be >= 1")
        if ic_mode not in ("canonical", "uniform"):
            raise ValueError("ic_mode must be 'canonical' or 'uniform'")
        super().__init__(f"FPUT (m={m})", 4 * m, m=int(m), omega=omega,
                         ic_mode=ic_mode, ic_noise=ic_noise, box=box)

    # ---- helpers -------------------------------------------------------------------------
    def _split(self, state):
        n = 2 * self.params["m"]
        return np.asarray(state[:n], dtype=float), np.asarray(state[n:], dtype=float)

    def _padded(self, q):
        return np.concatenate(([0.0], q, [0.0]))             # walls: q_0 = q_2m+1 = 0

    # ---- Hamiltonian system --------------------------------------------------------------
    def hamiltonian(self, state: np.ndarray) -> float:
        m, w = self.params["m"], self.params["omega"]
        q, p = self._split(state)
        Q = self._padded(q)
        d_stiff = Q[2:2 * m + 1:2] - Q[1:2 * m:2]            # (q_2i - q_2i-1), m springs
        d_soft = Q[1::2] - Q[0::2]                           # (q_2j+1 - q_2j), m+1 springs
        return 0.5 * np.sum(p ** 2) + w ** 2 / 4.0 * np.sum(d_stiff ** 2) + np.sum(d_soft ** 4)

    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        m, w = self.params["m"], self.params["omega"]
        q, p = self._split(state)
        Q = self._padded(q)
        d_stiff = Q[2:2 * m + 1:2] - Q[1:2 * m:2]
        d_soft = Q[1::2] - Q[0::2]
        grad = np.zeros_like(Q)                              # dV/dQ on the padded vector
        grad[2:2 * m + 1:2] += w ** 2 / 2.0 * d_stiff
        grad[1:2 * m:2] -= w ** 2 / 2.0 * d_stiff
        grad[1::2] += 4.0 * d_soft ** 3
        grad[0::2] -= 4.0 * d_soft ** 3
        return np.concatenate([p, -grad[1:-1]])              # [dq/dt, dp/dt] = [p, -dV/dq]

    def stiff_energies(self, state: np.ndarray) -> np.ndarray:
        """I_j = 1/2 ydot_j^2 + omega^2/2 y_j^2, y_j = (q_2j - q_2j-1)/sqrt2. Shape (m,)."""
        w = self.params["omega"]
        q, p = self._split(state)
        y, vy = (q[1::2] - q[0::2]) / np.sqrt(2), (p[1::2] - p[0::2]) / np.sqrt(2)
        return 0.5 * vy ** 2 + 0.5 * w ** 2 * y ** 2

    # ---- initial conditions --------------------------------------------------------------
    def _from_smooth(self, x, y, vx, vy):
        """(x, y, xdot, ydot) -> [q, p]; x_i=(q_2i-1+q_2i)/sqrt2, y_i=(q_2i-q_2i-1)/sqrt2."""
        s = np.sqrt(2.0)
        q, p = np.empty(2 * len(x)), np.empty(2 * len(x))
        q[0::2], q[1::2] = (x - y) / s, (x + y) / s
        p[0::2], p[1::2] = (vx - vy) / s, (vx + vy) / s
        return np.concatenate([q, p])

    def canonical_initial_state(self) -> np.ndarray:
        """GNI initial value: x_1 = 1, xdot_1 = 1, y_1 = 1/omega, ydot_1 = 1, all else 0
        (total energy H ~ 2; the first stiff spring holds I_1 = 1)."""
        m, w = self.params["m"], self.params["omega"]
        x, y, vx, vy = (np.zeros(m) for _ in range(4))
        x[0], vx[0], y[0], vy[0] = 1.0, 1.0, 1.0 / w, 1.0
        return self._from_smooth(x, y, vx, vy)

    def generate_initial_conditions(self, n_trajectories: int, seed: Optional[int] = None) -> np.ndarray:
        """ic_mode='canonical': the GNI initial state plus Gaussian noise of size ic_noise,
        applied in the smooth coordinates (y scaled by 1/omega, so the stiff-spring energy is
        perturbed by an amount comparable to the other modes). ic_noise=0 -> exact GNI state.

        ic_mode='uniform': q, p ~ U(-box, box) (hypercube sampling). With large omega these
        states carry huge stiff-spring energy (~omega^2); use omega of order 1 for this mode."""
        rng = np.random.default_rng(seed)
        m, w, dim = self.params["m"], self.params["omega"], self.state_dim
        if self.params["ic_mode"] == "uniform":
            b = self.params["box"]
            return rng.uniform(-b, b, size=(n_trajectories, dim))

        s = self.params["ic_noise"]
        base = self.canonical_initial_state()
        out = np.empty((n_trajectories, dim))
        for i in range(n_trajectories):
            x, y, vx, vy = (np.zeros(m) for _ in range(4))
            x[0], vx[0], y[0], vy[0] = 1.0, 1.0, 1.0 / w, 1.0
            x = x + s * rng.standard_normal(m)
            vx = vx + s * rng.standard_normal(m)
            y = y + s / w * rng.standard_normal(m)
            vy = vy + s * rng.standard_normal(m)
            out[i] = self._from_smooth(x, y, vx, vy)
        return out

    def get_plot_config(self) -> Dict:
        n = 2 * self.params["m"]
        return {
            "state_labels": [rf"$q_{{{i + 1}}}$" for i in range(n)] + [rf"$p_{{{i + 1}}}$" for i in range(n)],
            "phase_space": False,
            "x_range": [-2.1, 2.1],
            "y_range": [-2.1, 2.1],
        }
