import numpy as np


def verlet(f, tspan, y0, n):
    """Velocity-Verlet (Stormer-Verlet / leapfrog) for separable Hamiltonians H = T(p) + V(q).

    Explicit, symplectic, second order, ONE force evaluation per step (the force from the
    end of a step is reused in the next one). Use it for large systems where the implicit
    midpoint rule (a nonlinear solve per step) is too expensive.

    Works with any `f(t, y) -> [dq/dt, dp/dt]` for state y = [q, p] when dq/dt depends only on p
    and dp/dt only on q (e.g. pendulum, mass-spring, Henon-Heiles, N-body, Lennard-Jones).
    It is NOT valid for non-separable H such as the double pendulum.

    Returns t [n+1], y [n+1, 2*dof] like `midpoint`.
    """
    y0 = np.asarray(y0, dtype=float).ravel()
    h = len(y0) // 2
    dt = (tspan[1] - tspan[0]) / float(n)
    t = tspan[0] + dt * np.arange(n + 1)

    y = np.zeros((n + 1, len(y0)))
    y[0] = y0
    q, p = y0[:h].copy(), y0[h:].copy()
    force = f(t[0], y0)[h:]                                   # dp/dt = -dV/dq at q0
    for i in range(n):
        p_half = p + 0.5 * dt * force
        q = q + dt * f(t[i], np.concatenate([q, p_half]))[:h]  # dq/dt = dT/dp at p_half
        force = f(t[i + 1], np.concatenate([q, p_half]))[h:]
        p = p_half + 0.5 * dt * force
        y[i + 1, :h], y[i + 1, h:] = q, p
    return t, y
