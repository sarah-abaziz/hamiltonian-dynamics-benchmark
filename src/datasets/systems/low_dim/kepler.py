import numpy as np
from typing import Dict, Optional
from ...base import DynamicalSystem


class KeplerProblem(DynamicalSystem):
    """Kepler problem: one body in the inverse-square field of a fixed centre (unit mass).

    State [q (dim), p (dim)], dim = 2 (state_dim 4, planar) or 3 (state_dim 6).

        H = |p|^2 / 2  -  mu / |q|                      (separable: Verlet is valid)

    Besides H, the flow conserves the angular momentum L = q x p and the Runge-Lenz vector
    A = p x L - mu q/|q|  (see `invariants`). Numerical methods that are not symplectic show
    energy drift and a spurious precession of the orbit, i.e. a rotating A.

    The potential is singular at q = 0. Initial conditions are bound ellipses whose
    pericentre a(1 - e) is at least a_min (1 - ecc_max), so orbits stay away from the singularity.
    Orbits are sampled uniformly in mean anomaly (not in time, i.e. not biased to the apocentre).
    """

    default_integrator = "verlet"

    def __init__(self, dim: int = 2, mu: float = 1.0, a_range=(0.8, 1.2), ecc_max: float = 0.6):
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        if mu <= 0:
            raise ValueError("mu must be > 0")
        lo, hi = a_range
        if not 0 < lo <= hi:
            raise ValueError("a_range must satisfy 0 < lo <= hi")
        if not 0.0 <= ecc_max < 1.0:
            raise ValueError("ecc_max must be in [0, 1)")
        super().__init__(f"Kepler ({dim}D)", 2 * dim, dim=int(dim), mu=float(mu),
                         a_range=(float(lo), float(hi)), ecc_max=float(ecc_max))
        self.dim, self.mu = int(dim), float(mu)

    # ---- helpers -----------------------------------------------------------------------
    def _split(self, state):
        state = np.asarray(state, dtype=float)
        return state[..., : self.dim], state[..., self.dim:]

    def _to3(self, x):
        return x if self.dim == 3 else np.concatenate([x, np.zeros_like(x[..., :1])], axis=-1)

    # ---- Hamiltonian system ------------------------------------------------------------
    def hamiltonian(self, state: np.ndarray) -> float:
        q, p = self._split(state)
        H = 0.5 * np.sum(p ** 2, axis=-1) - self.mu / np.sqrt(np.sum(q ** 2, axis=-1))
        return H if np.ndim(H) else float(H)

    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        q, p = self._split(state)
        r3 = np.sum(q ** 2, axis=-1, keepdims=True) ** 1.5
        return np.concatenate([p, -self.mu * q / r3], axis=-1)

    # ---- invariants ----------------------------------------------------------------------
    def angular_momentum(self, state: np.ndarray) -> np.ndarray:
        """L = q x p. Shape (..., 1) in 2D (the z component), (..., 3) in 3D."""
        q, p = self._split(state)
        L = np.cross(self._to3(q), self._to3(p))
        return L[..., 2:3] if self.dim == 2 else L

    def runge_lenz(self, state: np.ndarray) -> np.ndarray:
        """A = p x L - mu q/|q|. Shape (..., 2) in 2D, (..., 3) in 3D. |A| = mu * eccentricity."""
        q, p = self._split(state)
        q3, p3 = self._to3(q), self._to3(p)
        r = np.sqrt(np.sum(q3 ** 2, axis=-1, keepdims=True))
        A = np.cross(p3, np.cross(q3, p3)) - self.mu * q3 / r
        return A[..., :2] if self.dim == 2 else A

    @property
    def invariant_names(self):
        if self.dim == 2:
            return ["H", "Lz", "Ax", "Ay"]
        return ["H", "Lx", "Ly", "Lz", "Ax", "Ay", "Az"]

    def invariants(self, state: np.ndarray) -> np.ndarray:
        """[H, L, A] concatenated along the last axis (names in `invariant_names`)."""
        H = np.asarray(self.hamiltonian(state))[..., None]
        return np.concatenate([H, self.angular_momentum(state), self.runge_lenz(state)], axis=-1)

    # ---- initial conditions --------------------------------------------------------------
    @staticmethod
    def _solve_kepler(M, e, iters: int = 60):
        E = M + e * np.sin(M)
        for _ in range(iters):
            E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        return E

    def generate_initial_conditions(self, n_trajectories: int, seed: Optional[int] = None) -> np.ndarray:
        """Bound elliptic orbits: semi-major axis a ~ U(a_range), eccentricity e ~ U(0, ecc_max),
        uniform mean anomaly, random orientation (a rotation in 2D, a random rotation in 3D)."""
        rng = np.random.default_rng(seed)
        n, mu = n_trajectories, self.mu
        a = rng.uniform(*self.params["a_range"], size=n)
        e = rng.uniform(0.0, self.params["ecc_max"], size=n)
        E = self._solve_kepler(rng.uniform(0.0, 2 * np.pi, size=n), e)
        r = a * (1.0 - e * np.cos(E))
        pos = np.stack([a * (np.cos(E) - e), a * np.sqrt(1 - e ** 2) * np.sin(E)], axis=1)
        vel = np.stack([-np.sqrt(mu * a) * np.sin(E) / r,
                        np.sqrt(mu * a * (1 - e ** 2)) * np.cos(E) / r], axis=1)
        if self.dim == 2:
            w = rng.uniform(0.0, 2 * np.pi, size=n)
            c, s = np.cos(w), np.sin(w)
            R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], axis=1)   # (n, 2, 2)
            q = np.einsum("nij,nj->ni", R, pos)
            p = np.einsum("nij,nj->ni", R, vel)
        else:
            q, p = np.empty((n, 3)), np.empty((n, 3))
            for i in range(n):
                Q, Rr = np.linalg.qr(rng.standard_normal((3, 3)))
                Q = Q * np.sign(np.diag(Rr))
                if np.linalg.det(Q) < 0:
                    Q[:, 0] *= -1
                q[i] = Q @ np.array([pos[i, 0], pos[i, 1], 0.0])
                p[i] = Q @ np.array([vel[i, 0], vel[i, 1], 0.0])
        return np.concatenate([q, p], axis=1)

    def get_plot_config(self) -> Dict:
        labels = [rf"$q_{{{i + 1}}}$" for i in range(self.dim)] + [rf"$p_{{{i + 1}}}$" for i in range(self.dim)]
        return {"state_labels": labels, "phase_space": False,
                "x_range": [-2.1, 2.1], "y_range": [-2.1, 2.1]}
