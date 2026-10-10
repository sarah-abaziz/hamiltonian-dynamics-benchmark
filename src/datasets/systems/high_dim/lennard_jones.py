import numpy as np
from typing import Optional
from ...base import DynamicalSystem


class LennardJonesFluid(DynamicalSystem):
    """Lennard-Jones fluid: N identical particles in a periodic cubic box (3D), NVE.

    Reduced units: sigma = epsilon = mass = 1 (time unit ~ 2.2 ps for argon).
    State [q (3N), p (3N)] with particle-major layout [x1,y1,z1,x2,...| px1,py1,pz1,...].

        H = sum |p_i|^2 / 2  +  sum_{i<j} V(r_ij),        r_ij by the minimum-image convention

    V is the *shifted-force* Lennard-Jones potential: V(r) = u(r) - u(rc) - u'(rc)(r - rc) for
    r < rc and 0 beyond. Both V and its force are continuous at the cutoff, so H is smooth,
    energy is conserved well, and the dynamics are exactly Hamilton's equations of this H.

    Positions are UNWRAPPED: an atom that crosses the box face keeps increasing its
    coordinate instead of reappearing on the other side (the force uses the nearest periodic
    image, so the physics is identical, and trajectories stay continuous like the pendulum
    angle). H is invariant under shifting any q_i by the box length L.

    Defaults are the classic liquid state: density 0.8442, T = 1.0, N = 4 n_cells^3 = 108.
    Initial conditions are equilibrated liquid snapshots (nearly equal total energy, zero total
    momentum). Use dt <= 0.005; the energy error of Verlet scales as dt^2.
    """

    default_integrator = "verlet"

    def __init__(self, n_cells: int = 3, density: float = 0.8442, temperature: float = 1.0,
                 cutoff: Optional[float] = None, sample_interval: int = 200,
                 equil_steps: int = 2000, equil_dt: float = 0.005):
        if n_cells < 2:
            raise ValueError("n_cells must be >= 2")
        n = 4 * n_cells ** 3
        L = float((n / density) ** (1.0 / 3.0))
        rc = min(2.5, 0.5 * L) if cutoff is None else float(cutoff)
        if rc > 0.5 * L:
            raise ValueError(f"cutoff {rc} must be <= L/2 = {0.5 * L:.3f} (minimum-image convention); "
                             "use a larger n_cells or a smaller cutoff.")
        super().__init__(f"Lennard-Jones fluid (N={n})", 6 * n, n_cells=int(n_cells),
                         n_particles=n, density=float(density), temperature=float(temperature),
                         box_length=L, cutoff=float(rc), sample_interval=int(sample_interval),
                         equil_steps=int(equil_steps), equil_dt=float(equil_dt))
        self.N, self.L, self.rc = n, L, float(rc)
        inv_rc = 1.0 / self.rc
        self._u_rc = 4.0 * (inv_rc ** 12 - inv_rc ** 6)
        self._du_rc = 4.0 * (-12.0 * inv_rc ** 13 + 6.0 * inv_rc ** 7)   # u'(rc)

    # ---- physics ---------------------------------------------------------------------------
    def _pairs(self, q):
        """Minimum-image displacement matrix and the mask of interacting pairs."""
        q = q.reshape(self.N, 3)
        dr = q[:, None, :] - q[None, :, :]
        dr -= self.L * np.round(dr / self.L)
        r2 = np.einsum("ijk,ijk->ij", dr, dr)
        mask = (r2 < self.rc ** 2) & ~np.eye(self.N, dtype=bool)
        return dr, np.where(mask, r2, 1.0), mask

    def potential_energy(self, q: np.ndarray) -> float:
        _, r2, mask = self._pairs(q)
        inv6 = r2 ** -3
        r = np.sqrt(r2)
        u = 4.0 * (inv6 ** 2 - inv6) - self._u_rc - self._du_rc * (r - self.rc)
        return float(0.5 * np.sum(np.where(mask, u, 0.0)))

    def forces(self, q: np.ndarray) -> np.ndarray:
        dr, r2, mask = self._pairs(q)
        inv6 = r2 ** -3
        r = np.sqrt(r2)
        coef = (24.0 * (2.0 * inv6 ** 2 - inv6) / r + self._du_rc) / r      # F_ij = coef * dr_ij
        coef = np.where(mask, coef, 0.0)
        return np.einsum("ij,ijk->ik", coef, dr).ravel()

    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        n = 3 * self.N
        return np.concatenate([state[n:], self.forces(state[:n])])

    def hamiltonian(self, state: np.ndarray) -> float:
        n = 3 * self.N
        return 0.5 * float(np.sum(state[n:] ** 2)) + self.potential_energy(state[:n])

    # ---- initial conditions ------------------------------------------------------------------
    def _fcc_lattice(self):
        c = self.params["n_cells"]
        basis = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
        cells = np.array([[i, j, k] for i in range(c) for j in range(c) for k in range(c)])
        pos = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3) * (self.L / c)
        return pos.ravel()

    def _maxwell_velocities(self, rng, T):
        v = rng.standard_normal((self.N, 3))
        v -= v.mean(axis=0)                                   # zero total momentum
        v *= np.sqrt(T * 3 * (self.N - 1) / np.sum(v ** 2))   # exact temperature (3N-3 dof)
        return v.ravel()

    def generate_initial_conditions(self, n_trajectories: int, seed: Optional[int] = None) -> np.ndarray:
        """Melt an FCC lattice at the target temperature, equilibrate, then take
        `n_trajectories` snapshots `sample_interval` steps apart from one NVE run.
        Snapshots have nearly the same total energy (one NVE run) and are wrapped into [0, L)."""
        rng = np.random.default_rng(seed)
        T = self.params["temperature"]
        dt = self.params["equil_dt"]
        n_eq, gap = self.params["equil_steps"], self.params["sample_interval"]

        q, p = self._fcc_lattice(), self._maxwell_velocities(rng, T)
        F = self.forces(q)
        samples = []
        for step in range(n_eq + n_trajectories * gap):
            p_half = p + 0.5 * dt * F                          # velocity Verlet, unit mass
            q = q + dt * p_half
            F = self.forces(q)
            p = p_half + 0.5 * dt * F
            if step < int(0.75 * n_eq) and step % 10 == 0:     # thermostat by rescaling, early only
                p *= np.sqrt(T * 3 * (self.N - 1) / np.sum(p ** 2))
            if step >= n_eq and (step - n_eq + 1) % gap == 0:
                samples.append(np.concatenate([q - self.L * np.floor(q / self.L), p]))
        return np.array(samples)
