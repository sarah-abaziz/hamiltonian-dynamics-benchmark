"""Uniform interface over the existing models (which have slightly different APIs).

predict_derivative(model, x) -> dx/dt of shape (batch, 2n) for MLP, HNN, DoubleHeadHNN,
CholeskyHNN and GeoHNN.  Use this inside any (symplectic) integrator or loss.
"""
import torch


def predict_derivative(model, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "get_dynamics"):          # HNN, DoubleHeadHNN, CholeskyHNN
        return model.get_dynamics(x)
    return model(x)                              # MLP and GeoHNN: forward() returns dx/dt
