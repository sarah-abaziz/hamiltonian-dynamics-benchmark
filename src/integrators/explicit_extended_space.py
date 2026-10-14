"""Explicit symplectic integrators for NON-separable Hamiltonians."""
import numpy as np

# Pyhamsys Implementation 

# def _run(f, tspan, y0, n, *, solver, omega, tol=1e-10, max_iter=100, **extra):
#     from pyhamsys import HamSys, solve_ivp_sympext
#     try:
#         from pyhamsys import Parameters
#     except ImportError:
#         from pyhamsys.pyhamsys import Parameters

#     y0 = np.asarray(y0, dtype=float).ravel()
#     h = (tspan[1] - tspan[0]) / float(n)
#     t_eval = tspan[0] + h * np.arange(n + 1)

#     try:
#         hs = HamSys(ndof=len(y0) // 2)
#     except TypeError:                       
#         hs = HamSys()
#     hs.y_dot = lambda t, y: np.asarray(f(t, y), dtype=float)

#     if n < 2:
#         raise ValueError("n must be >= 2")
    
#     step_req = (tspan[1] - tspan[0]) / (n - 1.5)
#     params = Parameters(step=step_req, solver=solver, omega=omega, tol=tol, max_iter=max_iter, **extra)
#     sol = solve_ivp_sympext(hs, tuple(tspan), y0, params, t_eval=t_eval)
#     if abs(sol.step - h) > 1e-9 * h:
#         raise RuntimeError(f"pyhamsys used step {sol.step}, expected {h}")

#     y = np.asarray(sol.y)                   
#     if y.shape != (len(y0), n + 1):
#         raise RuntimeError(f"unexpected sol.y shape {y.shape}, expected {(len(y0), n + 1)}")
#     return np.asarray(sol.t), y.T, sol


# # def pihajoki_unmixed(f, tspan, y0, n, solver="Verlet", **kw):
# #     t, y, _ = _run(f, tspan, y0, n, solver=solver, omega=0.0, **kw)   
# #     return t, y

# # def tao(f, tspan, y0, n, solver="Verlet", omega=10.0, **kw):
# #     t, y, _ = _run(f, tspan, y0, n, solver=solver, omega=omega, **kw)
# #     return t, y


# Pihajoki 
def pihajoki(f, tspan, y0, n, alpha_m=1.0, beta_m=0.0):
    y0 = np.asarray(y0, dtype=float).ravel()
    d = len(y0) // 2
    h = (tspan[1] - tspan[0]) / float(n)
    a, b = float(alpha_m), float(beta_m)

    def F(q, p):                                  
        r = np.asarray(f(0.0, np.concatenate([q, p])), dtype=float)
        return r[:d], r[d:]

    def H1(s, q, qt, p, pt):
        da, db = F(q, pt); return q, qt + s * da, p + s * db, pt

    def H2(s, q, qt, p, pt):
        da, db = F(qt, p); return q + s * da, qt, p, pt + s * db

    def M(q, qt, p, pt):
        return a * q + (1 - a) * qt, (1 - a) * q + a * qt, b * p + (1 - b) * pt, (1 - b) * p + b * pt

    q, qt, p, pt = y0[:d].copy(), y0[:d].copy(), y0[d:].copy(), y0[d:].copy()
    out = np.empty((n + 1, 2 * d)); out[0] = y0
    
    for k in range(n):
        s = M(q, qt, p, pt)                       
        s = H2(h / 2, *s); s = H1(h / 2, *s); s = M(*s); s = H1(h / 2, *s); s = H2(h / 2, *s)
        q, qt, p, pt = s
        # out[k + 1] = np.concatenate([(q + qt) / 2, (p + pt) / 2])
        out[k + 1] = np.concatenate([q, p])
        
    return tspan[0] + h * np.arange(n + 1), out


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
    """phi_2^delta = A(d/2) B(d/2) C(d) B(d/2) A(d/2)   (eq. 2).  Apply order: rightmost first."""
    z = flow_A(f, z, 0.5 * delta, d)
    z = flow_B(f, z, 0.5 * delta, d)
    z = flow_C(z, delta, omega, d)
    z = flow_B(f, z, 0.5 * delta, d)
    return flow_A(f, z, 0.5 * delta, d)


def make_step(order, literal_gamma=False):
    """Return step(f, z, delta, omega, d) of the requested even order via the triple jump (eq. 3)."""
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

def tao(f, tspan, y0, n, omega=20.0, order=2, literal_gamma=False, stats=None):
    """Tao's explicit symplectic integrator (symplectic in the extended phase space)."""
    
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
    for i in range(n):
        z = step(f, z, delta, omega, d)
        q, p, x, y = _split(z, d)
        Y[i + 1, :d], Y[i + 1, d:] = q, p
        X2[i + 1, :d], X2[i + 1, d:] = x, y
        defect[i + 1] = np.linalg.norm(np.concatenate([q - x, p - y]))
    if stats is not None:
        stats["defect"], stats["x"], stats["y"] = defect, X2[:, :d], X2[:, d:]
    return t, Y