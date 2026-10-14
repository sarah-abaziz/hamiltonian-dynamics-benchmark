import numpy as np
from typing import Dict, Optional
from ...base import DynamicalSystem


class ChargedParticleMagneticTrap(DynamicalSystem):
    """Charged particle (unit mass and charge) in a uniform magnetic field B e_z and a confining
    central potential V(r) = k r^2 / 2 + lam r^4 / 4.

    State [q (dim), p (dim)] with dim = 2 (state_dim 4) or 3 (state_dim 6). IMPORTANT: p is the
    CANONICAL momentum, not the velocity: in the symmetric gauge A(q) = (B/2)(-y, x, 0),

        pi = p - A(q)                       (mechanical momentum = velocity, dq/dt = pi)
        H  = |pi|^2 / 2 + V(|q|)           (non-separable: H contains the cross term p . A)

    so Verlet is invalid. Equations of motion:
        dq/dt = pi,    dp_x/dt = (B/2) pi_y - V'_x,    dp_y/dt = -(B/2) pi_x - V'_y,   dp_z/dt = -V'_z
    (equivalent to the Lorentz force: d pi_x/dt = B pi_y - V'_x, d pi_y/dt = -B pi_x - V'_y).

    V is axisymmetric, so the canonical angular momentum J_z = x p_y - y p_x is conserved
    (see `invariants`). Orbits are bounded because V is confining. The cyclotron frequency is B,
    so the step size must resolve 2 pi / B (and sqrt(k)).
    """

    default_integrator = "midpoint"

    def __init__(self, dim: int = 2, B: float = 2.0, k: float = 1.0, lam: float = 0.5,
                 box: float = 1.0, vmax: float = 1.0):
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        if k <= 0 or lam < 0:
            raise ValueError("need k > 0 and lam >= 0 for a confining potential")
        super().__init__(f"Charged particle in magnetic trap ({dim}D)", 2 * dim, dim=int(dim),
                         B=float(B), k=float(k), lam=float(lam), box=float(box), vmax=float(vmax))
        self.dim, self.B, self.k, self.lam = int(dim), float(B), float(k), float(lam)

    # ---- helpers -------------------------------------------------------------------------
    def _split(self, state):
        state = np.asarray(state, dtype=float)
        return state[..., : self.dim], state[..., self.dim:]

    def _vector_potential(self, q):
        A = np.zeros_like(q)
        A[..., 0] = -0.5 * self.B * q[..., 1]
        A[..., 1] = 0.5 * self.B * q[..., 0]
        return A

    def mechanical_momentum(self, state: np.ndarray) -> np.ndarray:
        """pi = p - A(q), which equals the velocity dq/dt."""
        q, p = self._split(state)
        return p - self._vector_potential(q)

    # ---- Hamiltonian system ------------------------------------------------------------
    def hamiltonian(self, state: np.ndarray) -> float:
        q, p = self._split(state)
        pi = p - self._vector_potential(q)
        r2 = np.sum(q ** 2, axis=-1)
        H = 0.5 * np.sum(pi ** 2, axis=-1) + 0.5 * self.k * r2 + 0.25 * self.lam * r2 ** 2
        return H if np.ndim(H) else float(H)

    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        q, p = self._split(state)
        pi = p - self._vector_potential(q)
        r2 = np.sum(q ** 2, axis=-1, keepdims=True)
        dp = -(self.k + self.lam * r2) * q                 # -grad V
        dp[..., 0] += 0.5 * self.B * pi[..., 1]
        dp[..., 1] -= 0.5 * self.B * pi[..., 0]
        return np.concatenate([pi, dp], axis=-1)

    # ---- invariants ----------------------------------------------------------------------
    @property
    def invariant_names(self):
        return ["H", "Jz"]

    def invariants(self, state: np.ndarray) -> np.ndarray:
        """[H, J_z] with J_z = x p_y - y p_x (canonical angular momentum about the field axis)."""
        q, p = self._split(state)
        Jz = q[..., 0] * p[..., 1] - q[..., 1] * p[..., 0]
        return np.stack([np.asarray(self.hamiltonian(state)), Jz], axis=-1)

    # ---- initial conditions --------------------------------------------------------------
    def generate_initial_conditions(self, n_trajectories: int, seed: Optional[int] = None) -> np.ndarray:
        """q ~ U(-box, box), velocity pi ~ U(-vmax, vmax); then p = pi + A(q), so the sampled
        quantity is the physical velocity and the energy range does not depend on B."""
        rng = np.random.default_rng(seed)
        q = rng.uniform(-self.params["box"], self.params["box"], size=(n_trajectories, self.dim))
        pi = rng.uniform(-self.params["vmax"], self.params["vmax"], size=(n_trajectories, self.dim))
        return np.concatenate([q, pi + self._vector_potential(q)], axis=1)

    def get_plot_config(self) -> Dict:
        labels = [rf"$q_{{{i + 1}}}$" for i in range(self.dim)] + [rf"$p_{{{i + 1}}}$" for i in range(self.dim)]
        return {"state_labels": labels, "phase_space": False,
                "x_range": [-2.0, 2.0], "y_range": [-2.0, 2.0]}
