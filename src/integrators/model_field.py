"""Bridge: trained torch model (HNN / GeoHNN) -> numpy f(t, y) used by the integrators."""
import numpy as np
import torch

from ..models.api import predict_derivative


def model_vector_field(model, device="cpu", dtype=torch.float64):
    """Return f(t, y) for ONE trajectory, y = [q, p] (numpy, shape (2d,))."""
    model = model.to(device=device, dtype=dtype).eval()

    def f(t, y):
        x = torch.as_tensor(np.asarray(y), dtype=dtype, device=device).reshape(1, -1)
        with torch.enable_grad():          # models differentiate H internally
            dx = predict_derivative(model, x)
        return dx.detach().cpu().numpy().ravel()

    return f


def batched_field(model, n_traj, dof, device="cpu", dtype=torch.float64):
    """f for N trajectories stacked as one big system, layout [q_all, p_all]."""
    model = model.to(device=device, dtype=dtype).eval()

    def f(t, y):
        y = np.asarray(y)
        q = y[: n_traj * dof].reshape(n_traj, dof)
        p = y[n_traj * dof:].reshape(n_traj, dof)
        x = torch.as_tensor(np.concatenate([q, p], 1), dtype=dtype, device=device)
        with torch.enable_grad():
            dx = predict_derivative(model, x).detach().cpu().numpy()
        return np.concatenate([dx[:, :dof].ravel(), dx[:, dof:].ravel()])

    return f


def rollout(model, integrator, y0, tspan, n, **kw):
    """Roll a model out with any integrator from src.integrators or src.datasets.integrators."""
    y0 = np.asarray(y0, dtype=float).ravel()
    return integrator(model_vector_field(model), tspan, y0, n, **kw)