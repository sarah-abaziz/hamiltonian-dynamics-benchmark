from functools import partial
from .explicit_extended_space import pihajoki, tao
from .semiexplicit_extended_space import semiexplicit
from .model_field import model_vector_field, rollout

INTEGRATORS = {
    "pihajoki": pihajoki,
    "tao": tao,
    "jayawardana_ohsawa": semiexplicit,                                            # order 2
    "jayawardana_ohsawa4": partial(semiexplicit, order=4, composition="suzuki"),   # order 4
}


def get_model_integrator(name):
    if name not in INTEGRATORS:
        raise ValueError(f"Unknown integrator '{name}'. Available: {sorted(INTEGRATORS)}")
    return INTEGRATORS[name]