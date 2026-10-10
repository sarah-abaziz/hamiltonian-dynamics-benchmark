"""Generate a dataset for a physical system.

Example:
    python -m scripts.generate_dataset --system simple_pendulum
    python -m scripts.generate_dataset --system double_pendulum --n-traj 200 --steps 1000
    python -m scripts.generate_dataset --system nbody_sun_earth_jupiter --tspan 0 4330 --steps 4330
    python -m scripts.generate_dataset --list
"""

import argparse
import ast

from src.datasets.generate import generate_dataset, save_dataset
from src.datasets.systems.low_dim import SimplePendulum, DoublePendulum, MassSpringSystem, HenonHeiles
from src.datasets.systems.high_dim import CoupledOscillators, presets, FermiPastaUlamTsingou, LennardJonesFluid
from src.datasets.integrators import INTEGRATORS


SYSTEMS = {
    "simple_pendulum": SimplePendulum,
    "double_pendulum": DoublePendulum,
    "mass_spring": MassSpringSystem,
    "henon_heiles": HenonHeiles,
    "coupled_oscillators": CoupledOscillators,
    "nbody_sun_earth": presets.sun_earth,
    "nbody_sun_earth_jupiter": presets.sun_earth_jupiter,
    "nbody_binary": presets.binary_star,
    "nbody_lagrange": presets.lagrange_triangle,
    "fput": FermiPastaUlamTsingou,
    "lennard_jones": LennardJonesFluid,
}


def parse_params(items, parser):
    """Parse parameters written as key=value."""
    params = {}
    for item in items:
        if "=" not in item:
            parser.error(f"Invalid parameter {item!r}. Use key=value, e.g. g=9.81")
        key, value = item.split("=", 1)
        try:
            params[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parser.error(f"Cannot parse value of {key!r}: {value!r}")
    return params


def main():
    parser = argparse.ArgumentParser(description="Generate Hamiltonian dynamics datasets.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--system", choices=SYSTEMS.keys())
    parser.add_argument("--integrator", choices=INTEGRATORS.keys(), default="midpoint")
    parser.add_argument("--n-traj", type=int, default=10)
    parser.add_argument("--tspan", type=float, nargs=2, default=[0.0, 10.0])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--params", nargs="*", default=[],
                        help="System parameters, e.g. g=9.81 L=1.0 m=1.0")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.list:
        print("Systems:", list(SYSTEMS.keys()))
        print("Integrators:", list(INTEGRATORS.keys()))
        return
    if args.system is None:
        parser.error("Specify --system or use --list.")
    if args.n_traj < 1 or args.steps < 1:
        parser.error("--n-traj and --steps must be positive.")
    if args.tspan[1] <= args.tspan[0]:
        parser.error("--tspan end must be greater than start.")

    params = parse_params(args.params, parser)
    try:
        system = SYSTEMS[args.system](**params)
    except TypeError as e:
        parser.error(f"Bad parameters for {args.system}: {e}")

    data = generate_dataset(system=system, n_trajectories=args.n_traj, tspan=tuple(args.tspan),
                            steps=args.steps, seed=args.seed, integrator=args.integrator,
                            system_key=args.system, verbose=True)

    t_end = f"{args.tspan[1]:g}".replace(".", "p")
    output = args.out or (f"data/generated/{args.system}__{args.integrator}"
                          f"__n{args.n_traj}_T{t_end}_s{args.steps}_seed{args.seed}.npz")

    saved_path = save_dataset(data, output)
    print(f"\nSaved: {saved_path}")
    print(f"States X shape: {data['X'].shape}")
    print(f"Derivatives dX shape: {data['dX'].shape}")
    print(f"Energy shape: {data['energy'].shape}")
    print(f"Metadata: {saved_path.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()