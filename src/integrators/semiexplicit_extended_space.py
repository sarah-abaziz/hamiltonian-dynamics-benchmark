"""Semiexplicit symplectic integrator for NON-separable Hamiltonians."""
import warnings
import numpy as np

# Yoshida's 6th-order composition (7 stages, solution A) built on the 2nd-order base.
_Y6 = (0.784513610477560, 0.235573213359357, -1.17767998417887)
_Y6 = _Y6 + (1.0 - 2.0 * sum(_Y6),) + _Y6[::-1]


# Explicit extended-space steps
def _flow_A(f, z, h, d):
    """Exact flow of H(q, y):  x += h D2H(q,y),  p -= h D1H(q,y)."""
    q, x, p, y = z[:d], z[d:2 * d], z[2 * d:3 * d], z[3 * d:]
    F = f(0.0, np.concatenate([q, y]))
    return np.concatenate([q, x + h * F[:d], p + h * F[d:], y])


def _flow_B(f, z, h, d):
    """Exact flow of H(x, p):  q += h D2H(x,p),  y -= h D1H(x,p)."""
    q, x, p, y = z[:d], z[d:2 * d], z[2 * d:3 * d], z[3 * d:]
    F = f(0.0, np.concatenate([x, p]))
    return np.concatenate([q + h * F[:d], x, p, y + h * F[d:]])


def _strang(f, z, h, d):
    """2nd-order Pihajoki step  A(h/2) o B(h) o A(h/2)."""
    z = _flow_A(f, z, 0.5 * h, d)
    z = _flow_B(f, z, h, d)
    return _flow_A(f, z, 0.5 * h, d)


def _compose(step, order, composition):
    """Build an explicit step of the requested (even) order from the Strang step."""
    if order == 2:
        return step
    if order % 2 or order < 2:
        raise ValueError("order must be an even integer >= 2")
    if composition == "yoshida":
        if order != 6:
            raise ValueError("composition='yoshida' is only available for order=6")
        base = step
        def yoshida(f, z, h, d):
            for w in _Y6:
                z = base(f, z, w * h, d)
            return z
        return yoshida

    lower = _compose(step, order - 2, composition)
    if composition == "triple_jump":
        a = 2.0 ** (1.0 / (order - 1))
        gs = (1.0 / (2.0 - a), -a / (2.0 - a), 1.0 / (2.0 - a))
    elif composition == "suzuki":
        a = 4.0 ** (1.0 / (order - 1))
        g, g3 = 1.0 / (4.0 - a), -a / (4.0 - a)
        gs = (g, g, g3, g, g)
    else:
        raise ValueError("composition must be 'triple_jump', 'suzuki' or 'yoshida'")

    def composed(f, z, h, d):
        for g in gs:
            z = lower(f, z, g * h, d)
        return z
    return composed


# Symmetric projection
def _AT(mu, d):
    """A^T mu = (mu1, -mu1, mu2, -mu2)."""
    return np.concatenate([mu[:d], -mu[:d], mu[d:], -mu[d:]])


def _A(z, d):
    """A z = (q - x, p - y)."""
    return np.concatenate([z[:d] - z[d:2 * d], z[2 * d:3 * d] - z[3 * d:]])


def semiexplicit(f, tspan, y0, n, order=2, composition="triple_jump", tol=1e-12,
                 max_iter=100, solver="newton", stats=None):

    if solver not in ("newton", "broyden"):
        raise ValueError("solver must be 'newton' or 'broyden'")
    y0 = np.asarray(y0, dtype=float).ravel()
    d = len(y0) // 2
    dt = (tspan[1] - tspan[0]) / float(n)
    t = tspan[0] + dt * np.arange(n + 1)
    phi = _compose(_strang, order, composition)

    def eval_f(zeta, mu):
        """zh = Phi_hat(zeta + A^T mu);  returns zh and f(mu) = A(zh + A^T mu)."""
        zh = phi(f, zeta + _AT(mu, d), dt, d)
        return zh, _A(zh + _AT(mu, d), d)

    Y = np.zeros((n + 1, 2 * d))
    Y[0] = y0
    iters, worst_defect, n_fail = [], 0.0, 0
    q, p = y0[:d].copy(), y0[d:].copy()

    for i in range(n):
        zeta = np.concatenate([q, q, p, p])                  # point on N
        mu = np.zeros(2 * d)
        zh, fv = eval_f(zeta, mu)
        H = np.eye(2 * d) / 4.0                              # (Df_0)^-1 = 1/4 I
        converged = False
        for k in range(max_iter):
            step = H @ fv if solver == "broyden" else 0.25 * fv
            if np.linalg.norm(step) < tol:                   # ||mu^(N+1) - mu^(N)|| < tol
                converged = True
                break
            mu_new = mu - step
            zh_new, fv_new = eval_f(zeta, mu_new)
            if solver == "broyden":                          # "good" Broyden, inverse update
                dm, df = mu_new - mu, fv_new - fv
                Hdf = H @ df
                denom = dm @ Hdf
                if abs(denom) > 1e-300:
                    H = H + np.outer(dm - Hdf, dm @ H) / denom
            mu, zh, fv = mu_new, zh_new, fv_new
        else:
            n_fail += 1
        iters.append(k)

        zeta_next = zh + _AT(mu, d)                          # lies on N up to tol
        q, p = zeta_next[:d].copy(), zeta_next[2 * d:3 * d].copy()
        Y[i + 1, :d], Y[i + 1, d:] = q, p
        if stats is not None:
            worst_defect = max(worst_defect, float(np.linalg.norm(_A(zeta_next, d))))

    if n_fail:
        warnings.warn(f"semiexplicit: projection did not reach tol={tol:g} in {max_iter} iterations "
                      f"on {n_fail}/{n} steps. Use a smaller dt, solver='broyden', or a larger tol.")
    if stats is not None:
        stats["iterations"] = iters
        stats["max_defect"] = worst_defect
        stats["n_not_converged"] = n_fail
    return t, Y
