from .midpoint import midpoint
from .verlet import verlet

INTEGRATORS = {"midpoint": midpoint, "verlet": verlet}


def get_integrator(name):
    if name not in INTEGRATORS:
        raise ValueError(
            f"Unknown integrator '{name}'. Available: {sorted(INTEGRATORS)}"
        )
    return INTEGRATORS[name]