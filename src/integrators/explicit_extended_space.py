"""Explicit / semi-explicit symplectic integrators for NON-separable Hamiltonians.

Wrappers around pyHamSys (https://github.com/cchandre/pyhamsys, BSD-2), written against the
pyhamsys 0.90: solve_ivp_sympext(hs, t_span, y0, params, t_eval) in pyhamsys.py.

In that source the variant is selected ONLY by params.omega (params.projection is not read):
    omega is a number  -> Tao restraint, no projection
    omega is None      -> Pihajoki + MIDPOINT projection (NOT plain Pihajoki)
    omega = 0.0        -> coupling is the identity, no projection -> plain Pihajoki
    
Signature:  f(t, y), tspan, y0, n -> t[n+1], y[n+1, 2d].
State layout is y = [q_1..q_d, p_1..p_d].  sol.y is the average of the two copies, (2d, n+1).
"""
import numpy as np


def _run(f, tspan, y0, n, *, solver, omega, tol=1e-10, max_iter=100, **extra):
    from pyhamsys import HamSys, solve_ivp_sympext
    try:
        from pyhamsys import Parameters
    except ImportError:
        from pyhamsys.pyhamsys import Parameters

    y0 = np.asarray(y0, dtype=float).ravel()
    h = (tspan[1] - tspan[0]) / float(n)
    t_eval = tspan[0] + h * np.arange(n + 1)

    try:
        hs = HamSys(ndof=len(y0) // 2)
    except TypeError:                       
        hs = HamSys()
    hs.y_dot = lambda t, y: np.asarray(f(t, y), dtype=float)

    if n < 2:
        raise ValueError("n must be >= 2")
    
    # pyhamsys 0.90's adjust_step uses nstep = (ceil(T/step)//n_eval)*n_eval + n_eval, i.e. it
    # always takes MORE steps than asked: step == t_eval spacing silently gives step/2.
    # Asking for T/(n-1.5) makes ceil(T/step) = n-1 < n_eval, which gives exactly nstep = n.
    step_req = (tspan[1] - tspan[0]) / (n - 1.5)
    params = Parameters(step=step_req, solver=solver, omega=omega, tol=tol, max_iter=max_iter, **extra)
    sol = solve_ivp_sympext(hs, tuple(tspan), y0, params, t_eval=t_eval)
    if abs(sol.step - h) > 1e-9 * h:
        raise RuntimeError(f"pyhamsys used step {sol.step}, expected {h}")

    y = np.asarray(sol.y)                   
    if y.shape != (len(y0), n + 1):
        raise RuntimeError(f"unexpected sol.y shape {y.shape}, expected {(len(y0), n + 1)}")
    return np.asarray(sol.t), y.T, sol


def pihajoki_unmixed(f, tspan, y0, n, solver="Verlet", **kw):
    t, y, _ = _run(f, tspan, y0, n, solver=solver, omega=0.0, **kw)   
    return t, y


def tao(f, tspan, y0, n, solver="Verlet", omega=10.0, **kw):
    t, y, _ = _run(f, tspan, y0, n, solver=solver, omega=omega, **kw)
    return t, y

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