from .explicit_extended_space import pihajoki, tao
from .model_field import model_vector_field, rollout

INTEGRATORS = {
    "pihajoki": pihajoki,
    "tao": tao,
}


def get_model_integrator(name):
    if name not in INTEGRATORS:
        raise ValueError(f"Unknown integrator '{name}'. Available: {sorted(INTEGRATORS)}")
    return INTEGRATORS[name]