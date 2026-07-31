from collections.abc import Callable
import numpy as np


def calculate_energy_range(
    mass_stopping_power_func: Callable[[np.ndarray], np.ndarray],
    density_g_per_cm3: float,
    initial_energy_mev: float,
    final_energy_mev: float = 0.0,
    n_points: int = 1000,
) -> float:
    """Calculate the particle range from the mass stopping power.

    Uses

        Δx = (1 / ρ) ∫ 1/S(E) dE

    where S(E) is the mass stopping power.

    Args:
        mass_stopping_power_func:
            Function returning the mass stopping power [MeV / (mg/cm²)]
            for an array of energies [MeV].
        density_g_per_cm3:
            Material density [g/cm³].
        initial_energy_mev:
            Initial particle energy [MeV].
        final_energy_mev:
            Final particle energy [MeV]. Defaults to 0 MeV.
        n_points:
            Number of integration points.

    Returns:
        Particle range [cm].
    """
    if initial_energy_mev < final_energy_mev:
        raise ValueError("initial_energy_mev must be larger than final_energy_mev.")

    energy = np.linspace(final_energy_mev, initial_energy_mev, n_points)
    stopping_power = np.asarray(mass_stopping_power_func(energy))

    if np.any(stopping_power <= 0):
        raise ValueError("Mass stopping power must be positive.")

    integral = np.trapezoid(1.0 / stopping_power, energy)

    return float(integral / (density_g_per_cm3 * 1e3))  # convert g/cm³ to mg/cm³
