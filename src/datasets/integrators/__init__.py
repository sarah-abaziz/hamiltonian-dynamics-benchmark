from .midpoint import midpoint

INTEGRATORS = {"midpoint": midpoint}

def get_integrator(name):
    if name not in INTEGRATORS:
        raise ValueError(f"Unknown integrator '{name}'. Available: {sorted(INTEGRATORS)}")
    return INTEGRATORS[name]