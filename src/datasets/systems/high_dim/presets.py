"""Ready-made N-body configurations (units: AU, solar masses, days; G = 2.959e-4)."""
import numpy as np
from .nbody import CelestialBody, NBodySystem

G = 2.959e-4


def _sun():
    return CelestialBody("Sun", 1.0, np.zeros(3), np.zeros(3), "yellow", "*", 15)


def _planet(name, mass, radius, color):
    v = np.sqrt(G * 1.0 / radius)                       # circular orbit around a 1 Msun star
    return CelestialBody(name, mass, np.array([radius, 0.0, 0.0]),
                         np.array([0.0, mass * v, 0.0]), color, "o", 8)


def sun_earth(**kw):
    return NBodySystem([_sun(), _planet("Earth", 3.0e-6, 1.0, "blue")], **kw)


def sun_earth_jupiter(**kw):
    return NBodySystem([_sun(), _planet("Earth", 3.0e-6, 1.0, "blue"),
                        _planet("Jupiter", 9.5e-4, 5.2, "orange")], **kw)


def binary_star(**kw):
    m, r = 1.0, 1.0
    v = np.sqrt(G * m * r / (2 * r) ** 2)               # separation = 2r
    s1 = CelestialBody("Star 1", m, np.array([r, 0, 0.0]), np.array([0, m * v, 0.0]), "red", "*", 10)
    s2 = CelestialBody("Star 2", m, np.array([-r, 0, 0.0]), np.array([0, -m * v, 0.0]), "blue", "*", 10)
    return NBodySystem([s1, s2], **kw)


def lagrange_triangle(**kw):
    m, R = 1.0 / 3.0, 1.0
    v = np.sqrt(G * m / (np.sqrt(3) * R))               # circular orbit of an equilateral triangle
    bodies = []
    for i in range(3):
        a = i * 2 * np.pi / 3
        bodies.append(CelestialBody(f"Body {i+1}", m,
                                    np.array([R * np.cos(a), R * np.sin(a), 0.0]),
                                    m * v * np.array([-np.sin(a), np.cos(a), 0.0])))
    return NBodySystem(bodies, **kw)
