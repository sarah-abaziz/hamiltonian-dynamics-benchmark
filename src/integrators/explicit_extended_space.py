"""Explicit symplectic integrators for NON-separable Hamiltonians."""
import numpy as np

# Pihajoki
def pihajoki_step(f, z, h, d, alpha_m=1.0, beta_m=0.0):
    """One step  [M, H2(h/2) H1(h/2) M H1(h/2) H2(h/2)]  on an extended state z = (q, p, x, y).

    H1, H2 are exact flows of H(q, y) and H(x, p). M is the mixing map; 
    it is symplectic only for (alpha, beta) in {(1, 1), (0, 0)}. Defaults to (1, 0) 
    for the best long-term energy behavior (Pihajoki 2015).
    """
    a, b = float(alpha_m), float(beta_m)
    q, p, qt, pt = z[:d], z[d:2 * d], z[2 * d:3 * d], z[3 * d:]

    def F(u, v):
        r = np.asarray(f(0.0, np.concatenate([u, v])), dtype=float)
        return r[:d], r[d:]

    def H1(s, q, qt, p, pt):
        da, db = F(q, pt); return q, qt + s * da, p + s * db, pt

    def H2(s, q, qt, p, pt):
        da, db = F(qt, p); return q + s * da, qt, p, pt + s * db

    def M(q, qt, p, pt):
        return a * q + (1 - a) * qt, (1 - a) * q + a * qt, b * p + (1 - b) * pt, (1 - b) * p + b * pt

    s = M(q, qt, p, pt)
    s = H2(h / 2, *s); s = H1(h / 2, *s); s = M(*s); s = H1(h / 2, *s); s = H2(h / 2, *s)
    q, qt, p, pt = s
    return np.concatenate([q, p, qt, pt])


def _output(z, d, output):
    """Physical state returned by an extended-space method: copy 1, copy 2 or their mean."""
    c1, c2 = z[..., :2 * d], z[..., 2 * d:]
    if output == "q":
        return c1
    if output == "x":
        return c2
    if output == "mean":
        return 0.5 * (c1 + c2)
    raise ValueError("output must be 'q' (copy 1), 'x' (copy 2) or 'mean'")


def pihajoki(f, tspan, y0, n, alpha_m=1.0, beta_m=0.0, output="q", stats=None):
    """Pihajoki's explicit extended-phase-space integrator.

    The returned trajectory is the physical projection `output` of the extended state
    ('q' = copy 1 [default, unchanged behaviour], 'x' = copy 2, 'mean').  stats (dict) receives Z (n+1, 4d, layout q,p,x,y) and gap (n+1,) = ||copy1-copy2||.
    """
    y0 = np.asarray(y0, dtype=float).ravel()
    d = len(y0) // 2
    h = (tspan[1] - tspan[0]) / float(n)
    z = np.concatenate([y0, y0])
    out = np.empty((n + 1, 2 * d)); out[0] = y0
    Z = np.empty((n + 1, 4 * d)); Z[0] = z
    for k in range(n):
        z = pihajoki_step(f, z, h, d, alpha_m, beta_m)
        Z[k + 1] = z
        out[k + 1] = _output(z, d, output)
    if stats is not None:
        stats["Z"] = Z
        stats["gap"] = np.linalg.norm(Z[:, :2 * d] - Z[:, 2 * d:], axis=1)
    return tspan[0] + h * np.arange(n + 1), out


def pihajoki_extended_H(H, z, d):
    """Hbar = H(q, y) + H(x, p) for z = (q, p, x, y)."""
    q, p, x, y = _split(z, d)
    return float(H(np.concatenate([q, y])) + H(np.concatenate([x, p])))


# Tao
def _split(z, d):
    return z[:d], z[d:2 * d], z[2 * d:3 * d], z[3 * d:]


def flow_A(f, z, delta, d):
    """phi_{H_A}: H_A = H(q, y)."""
    q, p, x, y = _split(z, d)
    F = f(0.0, np.concatenate([q, y]))            # [dH/dp, -dH/dq] evaluated at (q, y)
    return np.concatenate([q, p + delta * F[d:], x + delta * F[:d], y])


def flow_B(f, z, delta, d):
    """phi_{H_B}: H_B = H(x, p)."""
    q, p, x, y = _split(z, d)
    F = f(0.0, np.concatenate([x, p]))            # [dH/dp, -dH/dq] evaluated at (x, p)
    return np.concatenate([q + delta * F[:d], p, x, y + delta * F[d:]])


def flow_C(z, delta, omega, d):
    """phi_{omega H_C}: harmonic binding, an exact rotation of (q-x, p-y)."""
    q, p, x, y = _split(z, d)
    c, s = np.cos(2.0 * omega * delta), np.sin(2.0 * omega * delta)
    a, b = q - x, p - y                           # (alpha, beta)
    a2, b2 = c * a + s * b, -s * a + c * b        # R(delta) (alpha, beta)
    qm, pm = 0.5 * (q + x), 0.5 * (p + y)         # conserved by phi_C
    return np.concatenate([qm + 0.5 * a2, pm + 0.5 * b2, qm - 0.5 * a2, pm - 0.5 * b2])


def strang_step(f, z, delta, omega, d):
    """phi_2^delta = A(d/2) B(d/2) C(d) B(d/2) A(d/2).  Apply order: rightmost first."""
    z = flow_A(f, z, 0.5 * delta, d)
    z = flow_B(f, z, 0.5 * delta, d)
    z = flow_C(z, delta, omega, d)
    z = flow_B(f, z, 0.5 * delta, d)
    return flow_A(f, z, 0.5 * delta, d)


def make_step(order, literal_gamma=False):
    """Return step(f, z, delta, omega, d) of the requested even order via the triple jump."""
    if order < 2 or order % 2:
        raise ValueError("order must be an even integer >= 2")
    if order == 2:
        return strang_step
    lower = make_step(order - 2, literal_gamma)
    exponent = 1.0 / (order + 1) if literal_gamma else 1.0 / (order - 1)
    g = 1.0 / (2.0 - 2.0 ** exponent)

    def step(f, z, delta, omega, d):
        for coef in (g, 1.0 - 2.0 * g, g):       # all three factors commute in this order (symmetric)
            z = lower(f, z, coef * delta, omega, d)
        return z
    return step

def tao_extended_H(H, z, omega, d):
    """Hbar = H(q, y) + H(x, p) + omega * 1/2 (|q - x|^2 + |p - y|^2) for z = (q, p, x, y)."""
    q, p, x, y = _split(z, d)
    Hc = 0.5 * (np.sum((q - x) ** 2) + np.sum((p - y) ** 2))
    return float(H(np.concatenate([q, y])) + H(np.concatenate([x, p])) + omega * Hc)


def tao_step(f, z, delta, omega, d, order=2, literal_gamma=False):
    """One Tao step on an extended state z = (q, p, x, y) (exactly the step used inside `tao`)."""
    return make_step(order, literal_gamma)(f, z, delta, omega, d)


def tao(f, tspan, y0, n, omega=20.0, order=2, literal_gamma=False, output="q", stats=None):
    """Tao's explicit symplectic integrator (symplectic in the extended phase space).

    The returned trajectory is the physical projection `output` ('q' = copy 1 [default], 'x', 'mean');
    that projection is not itself a symplectic map of the original phase space.
    stats receives Z (n+1, 4d, layout q,p,x,y), gap/defect (n+1,) = ||copy1 - copy2||, x, y.
    """
    
    y0 = np.asarray(y0, dtype=float).ravel()
    d = len(y0) // 2
    delta = (tspan[1] - tspan[0]) / float(n)
    t = tspan[0] + delta * np.arange(n + 1)
    step = make_step(order, literal_gamma)

    z = np.concatenate([y0[:d], y0[d:], y0[:d], y0[d:]])        # (Q0, P0, Q0, P0)
    Y = np.zeros((n + 1, 2 * d))
    Y[0] = y0
    defect = np.zeros(n + 1)
    X2 = np.zeros((n + 1, 2 * d))
    X2[0] = y0
    Z = np.zeros((n + 1, 4 * d))
    Z[0] = z
    for i in range(n):
        z = step(f, z, delta, omega, d)
        q, p, x, y = _split(z, d)
        Y[i + 1] = _output(z, d, output)
        X2[i + 1, :d], X2[i + 1, d:] = x, y
        Z[i + 1] = z
        defect[i + 1] = np.linalg.norm(np.concatenate([q - x, p - y]))
    if stats is not None:
        stats["defect"], stats["gap"], stats["x"], stats["y"], stats["Z"] = defect, defect, X2[:, :d], X2[:, d:], Z
    return t, Y