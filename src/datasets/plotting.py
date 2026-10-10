"""Plot helpers. All functions return the figure(s) and never call plt.show()."""
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from .base import DynamicalSystem


def simulate_system(system: DynamicalSystem, initial_state, tspan=(0, 50), steps=5000, integrator=None):
    t, traj = system.simulate_trajectory(initial_state, tspan, steps, integrator)
    return t, traj, system.trajectory_energy(traj)


def plot_phase_space(system, n_traj=10, tspan=(0, 50), steps=5000, integrator=None):
    """First coordinate q_1 against its momentum p_1."""
    d = system.state_dim // 2
    fig = plt.figure(figsize=(8, 6))
    for ic in system.generate_initial_conditions(n_traj, seed=42):
        _, traj, _ = simulate_system(system, ic, tspan, steps, integrator)
        plt.plot(traj[:, 0], traj[:, d], alpha=0.6)
        plt.scatter(ic[0], ic[d], s=50, c="red", zorder=5)
    plt.xlabel("$q_1$"); plt.ylabel("$p_1$")
    plt.title(f"{system.name} - Phase Space"); plt.grid(True, alpha=0.3); plt.tight_layout()
    return fig


def plot_energy(system, n_traj=5, tspan=(0, 50), steps=5000, integrator=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for ic in system.generate_initial_conditions(n_traj, seed=42):
        t, _, energy = simulate_system(system, ic, tspan, steps, integrator)
        ax1.plot(t, energy, alpha=0.7)
        ax2.plot(t, (energy - energy[0]) / np.abs(energy[0]), alpha=0.7)
    ax1.set(xlabel="Time", ylabel="Energy", title="Energy vs Time")
    ax2.set(xlabel="Time", ylabel="Relative drift", title="Energy conservation")
    ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
    plt.suptitle(system.name); plt.tight_layout()
    return fig


def plot_trajectories(system, n_traj=3, tspan=(0, 50), steps=5000, integrator=None, max_dims=4):
    n_dims = min(system.state_dim, max_dims)
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 2 * n_dims), squeeze=False)
    axes = axes[:, 0]
    for ic in system.generate_initial_conditions(n_traj, seed=42):
        t, traj, _ = simulate_system(system, ic, tspan, steps, integrator)
        for i in range(n_dims):
            axes[i].plot(t, traj[:, i], alpha=0.7)
            axes[i].set_ylabel(f"State {i}"); axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time")
    plt.suptitle(f"{system.name} - Time Evolution"); plt.tight_layout()
    return fig


def plot_everything(system, n_traj=5, tspan=(0, 50), steps=5000, integrator=None):
    return (plot_phase_space(system, n_traj, tspan, steps, integrator),
            plot_energy(system, n_traj, tspan, steps, integrator),
            plot_trajectories(system, min(n_traj, 3), tspan, steps, integrator))


# ---------------- N-body ----------------
def _positions(system, traj):
    n = 3 * system.n_bodies
    return traj[:, :n].reshape(len(traj), system.n_bodies, 3)


def plot_3d_trajectories(system, tspan=(0, 365), steps=5000, integrator=None):
    _, traj, _ = simulate_system(system, system.initial_state, tspan, steps, integrator)
    pos = _positions(system, traj)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    for i, b in enumerate(system.bodies):
        ax.plot(pos[:, i, 0], pos[:, i, 1], pos[:, i, 2], label=b.name, color=b.color, linewidth=1.5)
        ax.scatter(*pos[0, i], color=b.color, s=b.markersize * 20, marker=b.marker)
    ax.set_xlabel("X (AU)"); ax.set_ylabel("Y (AU)"); ax.set_zlabel("Z (AU)")
    ax.set_title(f"{system.name} - 3D Trajectories"); ax.legend(); plt.tight_layout()
    return fig


def plot_2d_orbits(system, tspan=(0, 365), steps=5000, integrator=None):
    _, traj, _ = simulate_system(system, system.initial_state, tspan, steps, integrator)
    pos = _positions(system, traj)
    fig = plt.figure(figsize=(10, 10))
    for i, b in enumerate(system.bodies):
        plt.plot(pos[:, i, 0], pos[:, i, 1], label=b.name, color=b.color, linewidth=2)
        plt.scatter(pos[0, i, 0], pos[0, i, 1], color=b.color, s=b.markersize * 20, marker=b.marker)
    plt.xlabel("X (AU)"); plt.ylabel("Y (AU)"); plt.title(f"{system.name} - Orbital Plane")
    plt.axis("equal"); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    return fig


def plot_nbody_energy(system, tspan=(0, 365), steps=5000, integrator=None):
    t, _, energy = simulate_system(system, system.initial_state, tspan, steps, integrator)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(t, energy, linewidth=2)
    ax1.set(xlabel="Time (days)", ylabel="Total Energy", title="Energy vs Time")
    drift = (energy - energy[0]) / np.abs(energy[0])
    ax2.plot(t, drift, linewidth=2)
    ax2.set(xlabel="Time (days)", ylabel="Relative Energy Drift", title="Energy Conservation")
    ax2.text(0.05, 0.95, f"Max drift: {np.max(np.abs(drift)):.2e}", transform=ax2.transAxes, va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
    plt.suptitle(system.name); plt.tight_layout()
    return fig


def analyze_nbody(system, tspan=(0, 365), steps=5000, integrator=None):
    return (plot_3d_trajectories(system, tspan, steps, integrator),
            plot_2d_orbits(system, tspan, steps, integrator),
            plot_nbody_energy(system, tspan, steps, integrator))
