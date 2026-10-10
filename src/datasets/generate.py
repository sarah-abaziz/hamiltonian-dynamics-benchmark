"""Dataset generation: simulate trajectories, save arrays + a JSON metadata file."""
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from .base import DynamicalSystem


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"Not JSON serializable: {type(o)}")


def generate_dataset(system: DynamicalSystem, n_trajectories: int, tspan: tuple, steps: int,
                     seed: Optional[int] = None, integrator: Optional[str] = None,
                     integrator_kwargs: Optional[dict] = None, system_key: Optional[str] = None,
                     verbose: bool = False) -> Dict:
    """Simulate `n_trajectories` trajectories from random initial conditions.

    Returns a dict with
        t       (steps+1,)
        X       (n_traj, steps+1, state_dim)   states [q, p]
        dX      (n_traj, steps+1, state_dim)   exact time derivatives from Hamilton's equations
        energy  (n_traj, steps+1)
        meta    JSON-able dict (system, params, integrator, tspan, steps, seed, energy drift...)
    """
    integrator_kwargs = integrator_kwargs or {}
    used = integrator or system.default_integrator
    ics = system.generate_initial_conditions(n_trajectories, seed)

    X, dX, E = [], [], []
    t = None
    for i, ic in enumerate(ics):
        t, y = system.simulate_trajectory(ic, tspan, steps, integrator, **integrator_kwargs)
        X.append(y)
        dX.append(np.array([system.dynamics(tt, s) for tt, s in zip(t, y)]))
        E.append(system.trajectory_energy(y))
        if verbose:
            print(f"  trajectory {i + 1}/{n_trajectories}")
    X, dX, E = np.stack(X), np.stack(dX), np.stack(E)

    E0 = E[:, :1]
    abs_drift = np.abs(E - E0)
    meta = {
        "system_key": system_key,
        "integrator": used,
        "integrator_kwargs": integrator_kwargs,
        "tspan": list(tspan),
        "steps": steps,
        "dt": (tspan[1] - tspan[0]) / steps,
        "seed": seed,
        "n_trajectories": n_trajectories,
        "max_abs_energy_drift": float(abs_drift.max()),
        "max_rel_energy_drift": float((abs_drift / np.maximum(np.abs(E0), 1e-12)).max()),
    }
    return {"t": t, "X": X, "dX": dX, "energy": E, "meta": meta}


def save_dataset(data: Dict, path) -> Path:
    """Write <path>.npz (t, X, dX, energy) and <path>.meta.json."""
    path = Path(path)
    if path.suffix == ".npz":
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), t=data["t"], X=data["X"], dX=data["dX"], energy=data["energy"])
    with open(path.with_suffix(".meta.json"), "w") as f:
        json.dump(data["meta"], f, indent=2, default=_json_default)
    return path.with_suffix(".npz")


def load_dataset(path) -> Dict:
    path = Path(path)
    if path.suffix == ".npz":
        path = path.with_suffix("")
    arrs = np.load(path.with_suffix(".npz"))
    with open(path.with_suffix(".meta.json")) as f:
        meta = json.load(f)
    return {"t": arrs["t"], "X": arrs["X"], "dX": arrs["dX"], "energy": arrs["energy"], "meta": meta}


def split_dataset(data: Dict, test_fraction: float = 0.2, seed: int = 0):
    """Split by TRAJECTORY (never within a trajectory) into (train, test) dicts."""
    n = data["X"].shape[0]
    idx = np.random.default_rng(seed).permutation(n)
    n_test = int(round(n * test_fraction))
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    def take(ix):
        return {"t": data["t"], "X": data["X"][ix], "dX": data["dX"][ix],
                "energy": data["energy"][ix], "meta": data["meta"]}
    return take(train_idx), take(test_idx)
