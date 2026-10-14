import numpy as np
from typing import Dict, Optional
from ...base import DynamicalSystem


class SphericalPendulum(DynamicalSystem):
    """Spherical pendulum: a point mass on a rigid rod of length l, free to move on a sphere.

    Coordinates: theta = polar angle measured from the DOWNWARD vertical (theta = 0 is the
    stable equilibrium), phi = azimuth. State [theta, phi, p_theta, p_phi], dimension 4.

        H = p_theta^2 / (2 m l^2)  +  p_phi^2 / (2 m l^2 sin^2 theta)  -  m g l cos(theta)

    The kinetic term depends on the position theta (a configuration-dependent metric), so H is
    NOT separable and Verlet is invalid. phi is cyclic, so p_phi (= L_z) is conserved; note it
    is conserved EXACTLY by every integrator in these coordinates (dp_phi/dt = 0), so it is not
    a useful discriminating diagnostic here: use the energy, the phase-space area and the
    behaviour of theta.

    The coordinates are singular at the poles (theta = 0 and pi). With p_phi != 0 the centrifugal
    barrier keeps theta away from both poles, and initial conditions are rejection-sampled so
    that the whole orbit stays at least `pole_margin` radians away from them. Orbits with
    p_phi = 0 (planar swings through the pole) are deliberately excluded.

    The sign convention is the one of heyoka's spherical-pendulum example (-m g l cos theta);
    other references measure theta from the upward vertical and write +m g l cos theta.
    """

    default_integrator = "midpoint"

    def __init__(self, m: float = 1.0, length: float = 1.0, g: float = 1.0,
                 theta_range=(0.6, 2.2), ptheta_max: float = 1.0,
                 pphi_range=(0.5, 1.5), pole_margin: float = 0.3):
        if min(m, length, g) <= 0:
            raise ValueError("m, length and g must be > 0")
        if not 0.0 < pole_margin < np.pi / 2:
            raise ValueError("pole_margin must be in (0, pi/2)")
        if not 0.0 < pphi_range[0] <= pphi_range[1]:
            raise ValueError("pphi_range must satisfy 0 < lo <= hi (p_phi = 0 hits the pole)")
        super().__init__("Spherical pendulum", 4, m=float(m), length=float(length), g=float(g),
                         theta_range=tuple(theta_range), ptheta_max=float(ptheta_max),
                         pphi_range=tuple(pphi_range), pole_margin=float(pole_margin))
        self.m, self.l, self.g = float(m), float(length), float(g)
        self.inertia = self.m * self.l ** 2

    # ---- Hamiltonian system ------------------------------------------------------------
    def hamiltonian(self, state: np.ndarray) -> float:
        s = np.asarray(state, dtype=float)
        th, pth, pph = s[..., 0], s[..., 2], s[..., 3]
        H = (pth ** 2 + pph ** 2 / np.sin(th) ** 2) / (2 * self.inertia) - self.m * self.g * self.l * np.cos(th)
        return H if np.ndim(H) else float(H)

    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=float)
        th, pth, pph = s[..., 0], s[..., 2], s[..., 3]
        sn, cs = np.sin(th), np.cos(th)
        return np.stack([
            pth / self.inertia,                                                            # dtheta/dt
            pph / (self.inertia * sn ** 2),                                                # dphi/dt
            pph ** 2 * cs / (self.inertia * sn ** 3) - self.m * self.g * self.l * sn,      # dp_theta/dt
            np.zeros_like(th),                                                             # dp_phi/dt
        ], axis=-1)

    # ---- diagnostics -----------------------------------------------------------------------
    def invariants(self, state: np.ndarray) -> np.ndarray:
        """[H, L_z = p_phi]. L_z is trivially conserved in these coordinates (see class doc)."""
        s = np.asarray(state, dtype=float)
        return np.stack([np.asarray(self.hamiltonian(s)), s[..., 3]], axis=-1)

    invariant_names = ["H", "Lz"]

    def cartesian(self, state: np.ndarray) -> np.ndarray:
        """Bob position (x, y, z), z up: z = -l cos(theta). Shape (..., 3)."""
        s = np.asarray(state, dtype=float)
        th, ph = s[..., 0], s[..., 1]
        return self.l * np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), -np.cos(th)], axis=-1)

    def theta_extent(self, state: np.ndarray):
        """(theta_min, theta_max) of the orbit through `state`, from the turning points of the
        effective potential (roots of a cubic in c = cos theta). None if it cannot be bracketed."""
        th, pph = float(state[0]), float(state[3])
        E = self.hamiltonian(np.asarray(state, dtype=float))
        a, w = self.inertia, self.m * self.g * self.l
        roots = np.roots([-2 * a * w, -2 * a * E, 2 * a * w, 2 * a * E - pph ** 2])
        r = roots[np.abs(roots.imag) < 1e-9].real
        r = r[(r > -1.0) & (r < 1.0)]
        c0 = np.cos(th)
        lower, upper = r[r <= c0 + 1e-9], r[r >= c0 - 1e-9]
        if len(lower) == 0 or len(upper) == 0:
            return None
        return float(np.arccos(upper.min())), float(np.arccos(lower.max()))

    def _clear_of_poles(self, state) -> bool:
        ext = self.theta_extent(state)
        if ext is None:
            return False
        m = self.params["pole_margin"]
        return ext[0] >= m and ext[1] <= np.pi - m

    # ---- initial conditions --------------------------------------------------------------
    def generate_initial_conditions(self, n_trajectories: int, seed: Optional[int] = None) -> np.ndarray:
        """theta ~ U(theta_range), phi ~ U(0, 2 pi), p_theta ~ U(-ptheta_max, ptheta_max),
        |p_phi| ~ U(pphi_range) with a random sign; samples whose orbit would come closer than
        pole_margin to a pole are rejected."""
        rng = np.random.default_rng(seed)
        out = np.empty((n_trajectories, 4))
        for i in range(n_trajectories):
            for _ in range(10000):
                s = np.array([
                    rng.uniform(*self.params["theta_range"]),
                    rng.uniform(0.0, 2 * np.pi),
                    rng.uniform(-self.params["ptheta_max"], self.params["ptheta_max"]),
                    rng.choice([-1.0, 1.0]) * rng.uniform(*self.params["pphi_range"]),
                ])
                if self._clear_of_poles(s):
                    out[i] = s
                    break
            else:
                raise RuntimeError("could not sample an orbit clear of the poles; "
                                   "relax pole_margin or the sampling ranges")
        return out

    def get_plot_config(self) -> Dict:
        return {"state_labels": [r"$\theta$", r"$\varphi$", r"$p_\theta$", r"$p_\varphi$"],
                "phase_space": False,
                "x_range": [0.0, np.pi], "y_range": [-2.0, 2.0]}
