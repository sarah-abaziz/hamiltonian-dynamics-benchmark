"""Shared helpers for the extended-phase-space notebooks (A: true dynamics, B: learned dynamics)."""

import time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from src.datasets.integrators import midpoint, verlet
from src.integrators.explicit_extended_space import tao, pihajoki, tao_step, pihajoki_step
from src.integrators.semiexplicit_extended_space import semiexplicit, _compose, _strang, to_qpxy

# ----------------------------------------------------------------------------
# Flat batched representation: N trajectories -> [q_all, p_all]
# ----------------------------------------------------------------------------
def stack(Y, d):
    # (N, 2d) -> flat [q_all, p_all]
    Y = np.asarray(Y, float)
    return np.concatenate([Y[:, :d].ravel(), Y[:, d:].ravel()])


def unstack(flat_traj, N, d):
    # (T, 2Nd) -> (N, T, 2d)
    q = flat_traj[:, : N * d].reshape(-1, N, d)
    p = flat_traj[:, N * d :].reshape(-1, N, d)
    return np.concatenate([q, p], axis=2).transpose(1, 0, 2)


def oracle_field_batched(system, N, d):
    # true dynamics for N trajectories stacked as one big system
    def f(t, flat):
        q = flat[: N * d].reshape(N, d)
        p = flat[N * d :].reshape(N, d)
        out = np.empty((N, 2 * d))
        for i in range(N):
            out[i] = system.dynamics(t, np.concatenate([q[i], p[i]]))
        return np.concatenate([out[:, :d].ravel(), out[:, d:].ravel()])

    return f


def solve_reference(f_flat, Y0, d, T, n, rtol, atol, fallbacks=((1e-8, 1e-8),)):
    # tight DOP853 solve on the same time grid -> (N, n+1, 2d); loosen tolerances if it fails
    last = None
    for rt, at in ((rtol, atol), *fallbacks):
        sol = solve_ivp(
            f_flat, (0.0, T), stack(Y0, d), method="DOP853",
            t_eval=np.linspace(0.0, T, n + 1), rtol=rt, atol=at,
        )
        if sol.success:
            return unstack(sol.y.T, len(Y0), d)
        last = sol.message
    raise RuntimeError(last)


class Counted:
    # counts field evaluations (weight = number of trajectories one call evaluates)
    def __init__(self, f, weight=1):
        self.f, self.w, self.n = f, weight, 0

    def __call__(self, t, y):
        self.n += self.w
        return self.f(t, y)


def rk4(f, tspan, y0, n):
    h = (tspan[1] - tspan[0]) / n
    y = np.asarray(y0, float).copy()
    t = tspan[0]
    out = np.empty((n + 1, y.size))
    out[0] = y
    for i in range(n):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        out[i + 1] = y
    return tspan[0] + h * np.arange(n + 1), out


SINGLE = {"midpoint": midpoint, "verlet": verlet}  # run one trajectory at a time
BATCHED = {"rk4": rk4}

# Extended-space methods: every run also returns the full extended trajectory Z (layout q, p, x, y)
# and, for J-O, the projection-solver statistics.
EXT_NAMES = {"pihajoki", "pihajoki00", "tao", "tao4", "jayawardana_ohsawa", "jayawardana_ohsawa4"}
JO_NAMES = {"jayawardana_ohsawa", "jayawardana_ohsawa4"}


def integrate_ext(name, omega, f, y0, T, n, stats=None, **jo_override):
    # one entry point for every extended-space method -> physical trajectory (n+1, 2*d_flat) = copy 1
    if name in ("pihajoki", "pihajoki00"):
        a, b = (0.0, 0.0) if name == "pihajoki00" else (1.0, 0.0)
        return pihajoki(f, (0.0, T), y0, n, alpha_m=a, beta_m=b, stats=stats)[1]
    if name in ("tao", "tao4"):
        return tao(f, (0.0, T), y0, n, omega=omega, order=4 if name == "tao4" else 2, stats=stats)[1]
    if name in JO_NAMES:
        kw = {**JO_KW, **jo_override}
        return semiexplicit(f, (0.0, T), y0, n, order=4 if name.endswith("4") else 2, stats=stats, **kw)[1]
    raise KeyError(name)


def run_method(name, omega, Y0, d, T, n, f_flat, f_single):
    # -> dict(traj=(N, n+1, 2d), seconds, evals = field evaluations per trajectory-step,
    #         ext = dict(Z, stats) for extended-space methods else None)
    Y0 = np.asarray(Y0, float)
    N = len(Y0)
    t0 = time.perf_counter()
    try:
        ext = None
        if name in SINGLE:
            cs = Counted(f_single)
            traj = np.stack(
                [SINGLE[name](cs, (0.0, T), y0, n)[1] for y0 in Y0]
            )
            evals = cs.n
        else:
            cf = Counted(f_flat, weight=N)
            flat0 = stack(Y0, d)
            if name in EXT_NAMES:
                st = {}
                y = integrate_ext(name, omega, cf, flat0, T, n, stats=st)
                ext = dict(Z=st["Z"], stats=st)
            else:
                _, y = BATCHED[name](cf, (0.0, T), flat0, n)
            traj, evals = unstack(y, N, d), cf.n
        return dict(traj=traj, seconds=time.perf_counter() - t0, evals=evals / (N * n), ext=ext)
    except Exception as e:
        print(f"   [{name}] failed: {type(e).__name__}: {e}")
        return dict(traj=None, seconds=np.nan, evals=np.nan, ext=None)


def run_all(methods, *args):
    return {
        label: run_method(name, omega, *args)
        for label, (name, omega) in methods.items()
    }

# ----------------------------------------------------------------------------
# Extended-space tools.  Extended state z = (q, p, x, y); copy 1 = (q, p), copy 2 = (x, y).
# For N stacked trajectories every block has the flat batched layout, so Z (T, 4*N*d) is split per trajectory.
# ----------------------------------------------------------------------------
def ext_split(Z, N, d):
    # Z (T, 4Nd) -> dict of (N, T, d) arrays for q, p, x, y
    Z = np.atleast_2d(Z)
    Nd = N * d
    return {k: Z[:, i * Nd:(i + 1) * Nd].reshape(-1, N, d).transpose(1, 0, 2)
            for i, k in enumerate("qpxy")}


def ext_physical(parts, which):
    # the physical projection returned by the method: copy 1, copy 2 or their mean -> (N, T, 2d)
    c1 = np.concatenate([parts["q"], parts["p"]], axis=-1)
    c2 = np.concatenate([parts["x"], parts["y"]], axis=-1)
    return {"copy1": c1, "copy2": c2, "mean": 0.5 * (c1 + c2)}[which]


def _apply_H(H_fn, a, b):
    # H_fn(a, b) for arrays (N, T, d): H_fn maps (M, 2d) -> (M,)
    N, T, d = a.shape
    return H_fn(np.concatenate([a, b], axis=-1).reshape(-1, 2 * d)).reshape(N, T)


def ext_hamiltonian(H_fn, parts, omega, idx):
    # Hbar = H(q, y) + H(x, p) + omega * 1/2 (|q - x|^2 + |p - y|^2)   (omega = 0 for Pihajoki and J-O)
    q, p, x, y = (parts[k][:, idx] for k in "qpxy")
    Hb = _apply_H(H_fn, q, y) + _apply_H(H_fn, x, p)
    if omega:
        Hb = Hb + omega * 0.5 * (np.sum((q - x) ** 2, -1) + np.sum((p - y) ** 2, -1))
    return Hb


def ext_invariant(I_fn, parts, idx):
    # extended Noether charge I(q, p) + I(x, y): conserved by the exact sub-flows whenever I is conserved by H
    q, p, x, y = (parts[k][:, idx] for k in "qpxy")
    v = _apply_H(I_fn, q, p) + _apply_H(I_fn, x, y)
    return np.abs(v - v[:, :1])


def ext_step_fn(name, omega, f, d_flat):
    # one EXPLICIT extended step z -> z' in layout (q, p, x, y); for J-O this is the unprojected composition Phi_hat
    if name in ("pihajoki", "pihajoki00"):
        a, b = (0.0, 0.0) if name == "pihajoki00" else (1.0, 0.0)
        return lambda z, h: pihajoki_step(f, z, h, d_flat, a, b)
    if name in ("tao", "tao4"):
        order = 4 if name == "tao4" else 2
        return lambda z, h: tao_step(f, z, h, omega, d_flat, order=order)
    if name in JO_NAMES:
        phi = _compose(_strang, 4 if name.endswith("4") else 2, JO_KW["composition"])
        return lambda z, h: to_qpxy(phi(f, to_qpxy(z, d_flat), h, d_flat), d_flat)
    raise KeyError(name)


def ext_sample_states(ext, N, d, traj_idx, time_idx):
    # extended states (q, p, x, y) of single trajectories along a run
    parts = ext_split(ext["Z"], N, d)
    return [np.concatenate([parts[k][i, j] for k in "qpxy"]) for i in traj_idx for j in time_idx]


def ext_symp_defect(system, name, omega, h, states):
    # || M^T Jbar M - Jbar ||_F of the one-step EXTENDED map, Jbar = blockdiag(J, J); 0 = symplectic in the extended space
    try:
        d = system.state_dim // 2
        J = np.block([[np.zeros((d, d)), np.eye(d)], [-np.eye(d), np.zeros((d, d))]])
        Jbar = np.block([[J, np.zeros_like(J)], [np.zeros_like(J), J]])
        step = ext_step_fn(name, omega, oracle_field_batched(system, 1, d), d)
        F = lambda z: step(z, h)
        vals = []
        for z in states:
            M = fd_jacobian(F, z)
            vals.append(np.linalg.norm(M.T @ Jbar @ M - Jbar))
        return float(np.mean(vals))
    except Exception as e:
        print(f"   [{name}] extended symplecticity defect failed: {type(e).__name__}: {e}")
        return np.nan


def ext_reversibility_error(name, omega, Y0, d, T, n, f_flat):
    # EXTENDED reversibility: n steps, flip (p, y), n steps, flip (p, y) -> distance to the extended start state.
    # (the physical test in `reversibility_error` restarts from a diagonal state and so forgets the copy gap)
    try:
        Y0 = np.asarray(Y0, float)
        N, Nd, h = len(Y0), len(Y0) * d, T / n
        step = ext_step_fn(name, omega, f_flat, Nd)
        a = stack(Y0, d)
        z0 = np.concatenate([a[:Nd], a[Nd:], a[:Nd], a[Nd:]])
        flip = lambda z: np.concatenate([z[:Nd], -z[Nd:2 * Nd], z[2 * Nd:3 * Nd], -z[3 * Nd:]])
        z = z0.copy()
        for _ in range(n):
            z = step(z, h)
        z = flip(z)
        for _ in range(n):
            z = step(z, h)
        z = flip(z)
        P = ext_split(np.stack([z, z0]), N, d)
        diff = np.sqrt(sum(((P[k][:, 0] - P[k][:, 1]) ** 2).sum(-1) for k in "qpxy"))
        return float(np.median(_clean(diff)))
    except Exception as e:
        print(f"   [{name}] extended reversibility failed: {type(e).__name__}: {e}")
        return np.nan


# ----------------------------------------------------------------------------
# Momentum-like invariants, detected automatically on the reference solution
# ----------------------------------------------------------------------------
def candidate_invariants(d):
    # (M, 2d) states [q, p] -> (M,)
    c = {}
    if d >= 2:
        c["L12"] = lambda X: X[:, 0] * X[:, d + 1] - X[:, 1] * X[:, d]       # angular momentum in the (q1, q2) plane
    for i in range(d):
        c[f"p{i + 1}"] = (lambda i: lambda X: X[:, d + i])(i)                # a momentum component (cyclic coordinate)
    c["P_sum"] = lambda X: X[:, d:].sum(1)                                   # total linear momentum
    return c


def detect_invariants(ref, d, tol=INV_TOL):
    # keep the candidates that the tight reference solve conserves AND that are not trivially zero
    N, T, D = ref.shape
    X = ref.reshape(-1, D)
    found = []
    for nm, fn in candidate_invariants(d).items():
        v = fn(X).reshape(N, T)
        scale = float(np.max(np.abs(v)))
        if scale > 1e-3 and np.max(np.abs(v - v[:, :1])) < tol * (1.0 + scale):
            found.append(dict(name=nm, fn=fn))
    if any(f["name"].startswith("p") for f in found):                       # P_sum is redundant then
        found = [f for f in found if f["name"] != "P_sum"]
    return found


def ext_diagnostics(r, name, omega, N, d, n, H_fn=None, inv=None):
    # Extended-space view of one run (None for non-extended methods).
    ext = r.get("ext")
    if ext is None:
        return None
    parts = ext_split(ext["Z"], N, d)
    idx = np.unique(np.linspace(0, n, min(n + 1, EXT_STRIDE_MAX)).round().astype(int))
    c1, c2, cm = (ext_physical(parts, k) for k in ("copy1", "copy2", "mean"))
    out = dict(idx=idx, gap=np.linalg.norm(c1 - c2, axis=2), copy2=c2, mean=cm, stats=ext["stats"])
    if H_fn is not None:
        om = omega if name in ("tao", "tao4") else 0.0
        Hb = ext_hamiltonian(H_fn, parts, om, idx)
        out["Hbar_drift"] = np.abs(Hb - Hb[:, :1])
        out["drift_copy2"] = energy_drift(H_fn, c2[:, idx])
        out["drift_mean"] = energy_drift(H_fn, cm[:, idx])
    if inv is not None:
        out["mom_ext"] = ext_invariant(inv["fn"], parts, idx)
    return out

# ----------------------------------------------------------------------------
# Metrics. Per-trajectory arrays have shape (N, T); scalars are medians over trajectories.
# ----------------------------------------------------------------------------
def _clean(A):
    A = np.asarray(A, float)
    return np.where(np.isfinite(A), A, np.inf)


def state_error(traj, ref):
    with np.errstate(all="ignore"):
        return np.linalg.norm(traj - ref, axis=2)  # (N, T)


def energy_drift(H_fn, traj):
    # H_fn maps (M, 2d) -> (M,)
    N, T, D = traj.shape
    with np.errstate(all="ignore"):
        H = H_fn(traj.reshape(-1, D)).reshape(N, T)
        return np.abs(H - H[:, :1])  # (N, T)


def true_H_fn(system):
    return lambda X: np.array([float(system.hamiltonian(s)) for s in X])


def err_at(A, i):
    return float(np.median(_clean(A)[:, i]))


def mean_over_time(A):
    with np.errstate(all="ignore"):
        return float(np.median(_clean(A).mean(axis=1)))


def max_over_time(A):
    return float(np.median(_clean(A).max(axis=1)))


def drift_slope(t, A, tmin_frac=0.1):
    # log-log slope of median drift vs t: ~0 bounded, ~1 linear, >1 faster growth
    med = np.median(_clean(A), axis=0)
    m = (t > tmin_frac * t[-1]) & np.isfinite(med) & (med > 1e-14)
    if m.sum() < 5:
        return np.nan
    return float(np.polyfit(np.log(t[m]), np.log(med[m]), 1)[0])


def fit_order(h, e, k=4):
    # observed order from the k smallest step sizes with a usable error
    h, e = np.asarray(h, float), np.asarray(e, float)
    m = np.isfinite(e) & (e > 1e-12)
    h, e = h[m], e[m]
    if len(h) < 3:
        return np.nan
    idx = np.argsort(h)[:k]
    return float(np.polyfit(np.log(h[idx]), np.log(e[idx]), 1)[0])


def fd_jacobian(F, y, eps=1e-5):
    # central finite differences; noise floor ~ (solver tolerance) / eps
    y = np.asarray(y, float)
    J = np.empty((len(y), len(y)))
    for k in range(len(y)):
        e = np.zeros(len(y))
        e[k] = eps
        J[:, k] = (F(y + e) - F(y - e)) / (2 * eps)
    return J


def one_step_map(system, name, omega, h):
    # physical one-step map y -> y(h) (copy 1 for extended methods) for a single trajectory started on the diagonal
    d = system.state_dim // 2
    if name in SINGLE:
        return lambda y: SINGLE[name](system.dynamics, (0.0, h), np.asarray(y, float), 1)[1][-1]
    f1 = oracle_field_batched(system, 1, d)
    if name in EXT_NAMES:
        step = lambda y: integrate_ext(name, omega, f1, stack(y[None], d), h, 1)
    else:
        step = lambda y: BATCHED[name](f1, (0.0, h), stack(y[None], d), 1)[1]
    return lambda y: unstack(step(np.asarray(y, float)), 1, d)[0, -1]


def sympl_defect(system, name, omega, h, Y_samples):
    # || M^T J M - J ||_F of the PHYSICAL one-step map in the original (q, p) space (0 = symplectic)
    try:
        d = system.state_dim // 2
        Jm = np.block([[np.zeros((d, d)), np.eye(d)], [-np.eye(d), np.zeros((d, d))]])
        F = one_step_map(system, name, omega, h)
        vals = []
        for y in Y_samples:
            M = fd_jacobian(F, y)
            vals.append(np.linalg.norm(M.T @ Jm @ M - Jm))
        return float(np.mean(vals))
    except Exception as e:
        print(f"   [{name}] symplecticity defect failed: {type(e).__name__}: {e}")
        return np.nan


def reversibility_error(name, omega, Y0, d, T, n, f_flat, f_single):
    # forward n steps, flip p, n steps again, flip p -> distance to the start (0 = reversible)
    Y0 = np.asarray(Y0, float)
    r = run_method(name, omega, Y0, d, T, n, f_flat, f_single)
    if r["traj"] is None:
        return np.nan
    Y1 = r["traj"][:, -1].copy()
    Y1[:, d:] *= -1
    r2 = run_method(name, omega, Y1, d, T, n, f_flat, f_single)
    if r2["traj"] is None:
        return np.nan
    Y2 = r2["traj"][:, -1].copy()
    Y2[:, d:] *= -1
    with np.errstate(all="ignore"):
        e = np.linalg.norm(Y2 - Y0, axis=1)
    return float(np.median(_clean(e)))


def phys_invariant_drift(I_fn, traj, idx):
    # |I(t) - I(0)| of a physical trajectory (N, T, 2d) at the time indices idx -> (N, len(idx))
    N, T, D = traj.shape
    v = I_fn(traj[:, idx].reshape(-1, D)).reshape(N, len(idx))
    return np.abs(v - I_fn(traj[:, 0])[:, None])


def cost_to_reach(w, e, target):
    # smallest cost at which the error drops to `target` (log-log interpolation); NaN if never reached
    w, e = np.asarray(w, float), np.asarray(e, float)
    ok = np.isfinite(w) & np.isfinite(e) & (e > 0)
    if ok.sum() < 2:
        return np.nan
    w, e = w[ok], e[ok]
    o = np.argsort(w)
    w, e = w[o], e[o]
    hit = np.where(e <= target)[0]
    if len(hit) == 0:
        return np.nan
    j = hit[0]
    if j == 0:
        return float(w[0])
    lw = np.interp(np.log(target), [np.log(e[j]), np.log(e[j - 1])], [np.log(w[j]), np.log(w[j - 1])])
    return float(np.exp(lw))

# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def band(ax, t, A, label, color=None, ls=None):
    # median curve with inter-quartile band over trajectories (log y)
    A = _clean(A)
    with np.errstate(all="ignore"):
        m = np.median(A, 0)
        lo, hi = np.percentile(A, 25, 0), np.percentile(A, 75, 0)
    ok = np.isfinite(m) & (m < 1e30)
    ok[0] = False
    if not ok.any():
        return
    st = sty(label)
    if color is not None:
        st["color"] = color
    if ls is not None:
        st["ls"] = ls
    diverged = not np.isfinite(m[1:]).all()
    ax.semilogy(t[ok], m[ok] + 1e-16, label=label + (" (diverged)" if diverged else ""), **st)
    f = ok & np.isfinite(lo) & np.isfinite(hi) & (lo < 1e30) & (hi < 1e30)
    if f.any():
        ax.fill_between(t[f], lo[f] + 1e-16, hi[f] + 1e-16, color=st["color"], alpha=0.12, lw=0)


def heat(ax, piv, title, log=True, lower_is_better=True):
    M = piv.to_numpy(float)
    with np.errstate(all="ignore"):
        V = np.log10(np.where(M <= 0, 1e-17, M)) if log else M.copy()   # exact zeros are drawn at 1e-17
    V = np.where(np.isfinite(V), V, np.nan)
    ax.set_title(title)
    ax.set_xlabel(piv.columns.name or "")
    ax.set_ylabel(piv.index.name or "")
    ax.grid(False)
    ax.set_xticks(range(M.shape[1]), list(piv.columns), rotation=25, ha="right")
    ax.set_yticks(range(M.shape[0]), list(piv.index))
    if np.isnan(V).all():
        return
    im = ax.imshow(V, cmap="viridis_r" if lower_is_better else "viridis", aspect="auto")
    thr = np.nanmedian(V)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(V[i, j]):
                s, c = "n/a", "black"
            else:
                s = ("0" if M[i, j] == 0 else f"{M[i, j]:.0e}") if log else f"{M[i, j]:.2f}"
                c = "white" if (V[i, j] > thr) == lower_is_better else "black"
            ax.text(j, i, s, ha="center", va="center", fontsize=8, color=c)
    plt.colorbar(im, ax=ax, label="log10(value)" if log else "value")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")


# Axis labels with units (all quantities are in the system's own units; time, energy, ...)
LBL = dict(
    t=r"time $t$",
    drift=r"energy drift $|H(t)-H(0)|$",
    err=r"state error $\|(q,p)-(q,p)_{\mathrm{exact}}\|$",
    Hbar=r"extended-$H$ drift $|\bar{H}(t)-\bar{H}(0)|$",
    gap=r"copy separation $\|(q,p)-(x,y)\|$",
    h=r"step size $h$ [time units]",
    cost="field evaluations per unit time",
    sec="wall-clock time [s]",
    mom=r"momentum drift $|I(t)-I(0)|$",
)


def piv(df, col, rows="system", cols="method", ridx=None, rcol=None):
    # pivot a long result table; defaults to the systems / methods of Part A (SYS_OK, LAB_OK)
    p = df.pivot(index=rows, columns=cols, values=col)
    return p.reindex(index=ridx if ridx is not None else SYS_OK,
                     columns=rcol if rcol is not None else LAB_OK)