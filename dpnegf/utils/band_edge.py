from typing import Tuple, Optional

import numpy as np


def calculate_band_edges(
    eigenvalues: np.ndarray,
    neutral_electrons: float,
    spin_degeneracy: int,
    occupation_tolerance: float = 1e-8,
    gap_tolerance: float = 1e-6,
) -> Tuple[float, float]:
    """Return the conduction- and valence-band edges ``(E_c, E_v)``.

    The band indices are fixed by the neutral electron count, independently of
    the Fermi level or any finite-temperature smearing.  ``eigenvalues`` must
    have shape ``(n_kpoints, n_bands)``.
    """

    eigenvalues = np.asarray(eigenvalues)
    if eigenvalues.ndim != 2:
        raise ValueError(
            "eigenvalues must have shape (n_kpoints, n_bands), "
            f"got {eigenvalues.shape}."
        )
    if eigenvalues.shape[0] == 0 or eigenvalues.shape[1] == 0:
        raise ValueError("eigenvalues must contain at least one k-point and band.")
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("eigenvalues contain non-finite values.")
    if spin_degeneracy not in (1, 2):
        raise ValueError(
            f"spin_degeneracy must be 1 or 2, got {spin_degeneracy}."
        )

    occupied_bands_float = float(neutral_electrons) / spin_degeneracy
    occupied_bands = int(round(occupied_bands_float))
    if not np.isclose(
        occupied_bands_float, occupied_bands, rtol=0.0, atol=occupation_tolerance
    ):
        raise ValueError(
            "The neutral electron count does not define an integer number of "
            f"occupied bands: neutral_electrons={neutral_electrons}, "
            f"spin_degeneracy={spin_degeneracy}, "
            f"n_occupied={occupied_bands_float}."
        )

    n_bands = eigenvalues.shape[1]
    if not 1 <= occupied_bands < n_bands:
        raise ValueError(
            "The occupied-band index is outside the available spectrum: "
            f"n_occupied={occupied_bands}, n_bands={n_bands}."
        )

    sorted_eigenvalues = np.sort(eigenvalues, axis=1)

    e_v = float(np.max(sorted_eigenvalues[:, occupied_bands - 1]))
    e_c = float(np.min(sorted_eigenvalues[:, occupied_bands]))
    band_gap = e_c - e_v

    if band_gap <= gap_tolerance:
        raise ValueError(
            "The sampled spectrum does not have a band gap greater than "
            f"gap_tolerance={gap_tolerance}: "
            f"E_v={e_v}, E_c={e_c}, E_g={band_gap}."
        )

    return e_c, e_v

def validate_fermi_in_band_gap(
        e_fermi: float,
        e_c: float,
        e_v: float,
        lead_name: Optional[str] = None,
) -> None:
    """Raise an error if the Fermi level is not in the band gap."""

    label = f" for lead '{lead_name}'" if lead_name else ""
    if not (e_v < e_fermi < e_c):
        raise ValueError(
            f"The Fermi level is not in the band gap{label}: "
            f"E_v={e_v}, E_c={e_c}, E_F={e_fermi}."
        )