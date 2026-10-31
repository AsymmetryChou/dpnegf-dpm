"""Perfect (5,5) CNT: nearest-neighbor pi model, one orbital per carbon.

Run from the repository root:
    python -m examples.transmission_eigenchannels.run_cnt55 --blocks 100

Builds the actual ASE CNT geometry and its periodic H/S blocks, uses DPNEGF
surface self-energies and full RGF, and plots a spectrum and two complex channels.
This is an explicit pi model, independent of the fitted sp3 example model.
"""
import argparse
from pathlib import Path
import resource
import time

import numpy as np
import torch
from ase.build import nanotube
from ase.neighborlist import neighbor_list
from ase.io import write

from dpnegf.negf.recursive_green_cal import recursive_gf
from dpnegf.negf.surface_green import selfEnergy
from dpnegf.negf.trans_eigs import (
    TransportContext, transmission_eigenvalues, transmission_eigenchannels,
)


def cnt_model(overlap=.1):
    """Periodic pi model; half filling has E_F=0 eV (no onsite terms)."""
    unit = nanotube(5, 5, length=1, bond=1.42)
    unit.center(vacuum=10, axis=(0, 1))
    i, j, shifts = neighbor_list('ijS', unit, 1.6)
    n = len(unit)
    h0 = torch.zeros(n, n, dtype=torch.complex128)
    h1 = torch.zeros_like(h0)
    s0 = torch.eye(n, dtype=h0.dtype)
    s1 = torch.zeros_like(h0)
    for a, b, shift in zip(i, j, shifts):
        if shift[2] == 0:
            h0[a, b], s0[a, b] = -2.7, overlap
        elif shift[2] == 1:
            h1[a, b], s1[a, b] = -2.7, overlap
    return unit, h0, h1, s0, s1


def count_band_crossings(model, energy):
    """Independent generalized Bloch-band check of the propagating modes."""
    _, h0, h1, s0, s1 = model
    # Confirm this energy is inside two bands by counting crossings of the
    # periodic generalized eigenvalues, independently of transmission SVD.
    bands = []
    for k in np.linspace(-np.pi, np.pi, 1001):
        phase = np.exp(1j * k)
        H, S = h0 + phase * h1 + phase.conjugate() * h1.mH, s0 + phase * s1 + phase.conjugate() * s1.mH
        L = torch.linalg.cholesky(S)
        ortho = torch.linalg.solve_triangular(L, H, upper=False)
        ortho = torch.linalg.solve_triangular(L, ortho.mH, upper=False).mH
        bands.append(torch.linalg.eigvalsh(ortho))
    bands = torch.stack(bands)
    if abs(energy) < .02:
        # Sorted bands touch at the Dirac point; a finite k mesh misses that
        # zero-width crossing at E_F. Check both nearby energy surfaces instead.
        counts = [int(((bands[:-1] < e) & (bands[1:] >= e)).sum()) for e in (-.02, .02)]
        if counts[0] != counts[1]:
            raise ValueError('Inconsistent mode count on the two sides of the Dirac point')
        return counts[0]
    return int(((bands[:-1] < energy) & (bands[1:] >= energy)).sum())


def context_from_model(model, nblocks, energy, eta_lead=1e-5, eta_device=1e-8):
    unit, h0, h1, s0, s1 = model
    n = len(unit)
    # surface_green(h01) is the surface extending toward the positive index.
    right, _ = selfEnergy(h0, h1, s0, s1, energy, hDL=h1, sDL=s1,
                          etaLead=eta_lead, numba_jit=False)
    left, _ = selfEnergy(h0, h1.mH.resolve_conj(), s0, s1.mH.resolve_conj(), energy,
                         hDL=h1.mH.resolve_conj(), sDL=s1.mH.resolve_conj(),
                         etaLead=eta_lead, numba_jit=False)
    context = TransportContext(energy, [h0] * nblocks, [s0] * nblocks,
                               [h1] * (nblocks - 1), [h1.mH] * (nblocks - 1),
                               [s1] * (nblocks - 1), [s1.mH] * (nblocks - 1),
                               left, right, eta_device=eta_device, eta_lead=eta_lead, kpoint=[0., 0., 0.],
                               # Repeated unit atoms and H/S blocks share this explicit order.
                               orbital_map=torch.arange(n * nblocks))
    return context, unit.repeat((1, 1, nblocks))


def cnt_context(nblocks=10, energy=.2, overlap=.1):
    """Identical periodic device and leads; hopping -2.7 eV, nearest neighbors."""
    model = cnt_model(overlap)
    crossings = count_band_crossings(model, energy)
    if crossings != 2:
        raise ValueError(f'Chosen energy has {crossings} positive band crossings; expected two')
    return context_from_model(model, nblocks, energy)


def full_rgf(context):
    c = context
    full = recursive_gf(c.energy, c.hl, c.hd, c.hu, c.sd, c.su, c.sl,
                        c.left_se, c.right_se, eta=c.eta_device, keep_gr_left=True)
    c.gr_left = full[5]


def scan_spectrum(model, nblocks, energies, eta_lead=1e-5, eta_device=1e-8):
    """Keep scalar spectra only; each energy's Green blocks are released."""
    spectrum = []
    for energy in energies:
        c, _ = context_from_model(model, nblocks, float(energy), eta_lead, eta_device)
        full_rgf(c)
        result = transmission_eigenvalues(c)
        tau = result['tau_all'].detach().cpu().numpy()
        top = np.pad(tau[:2], (0, max(0, 2 - len(tau))))
        spectrum.append([energy, *top, float(result['T_spectral']),
                         float(result['T_caroli_raw'])])
        del result, c
    return np.asarray(spectrum)


def plot_results(output, spectrum, atoms, channels):
    # A noninteractive backend also works on compute nodes without a display.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout='constrained')
    ax.plot(spectrum[:, 0], spectrum[:, 1], lw=2.5, label=r'$\tau_1$ (largest)')
    ax.plot(spectrum[:, 0], spectrum[:, 2], '--', marker='o', markersize=4,
            markevery=max(1, len(spectrum) // 16), lw=1.5,
            label=r'$\tau_2$ (second largest)')
    ax.axvline(0, color='0.4', ls=':', label=r'$E_F$')
    ax.set(xlabel=r'$E-E_F$ (eV)', ylabel='Transmission eigenvalue',
           ylim=(-.03, 1.08), title='Perfect (5,5) CNT: two leading eigenvalues')
    ax.grid(alpha=.25)
    ax.legend(loc='lower center')
    fig.savefig(output / 'transmission_eigenvalues.png', dpi=180)
    plt.close(fig)

    xyz = atoms.positions.copy()
    xyz[:, :2] -= xyz[:, :2].mean(axis=0)
    theta = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2 * np.pi)
    coefficients = channels['coefficients'].detach().cpu().numpy()
    # Fix only a display phase; the saved complex coefficients stay untouched.
    references = np.argmax(np.abs(coefficients), axis=0)
    display = coefficients * np.exp(-1j * np.angle(coefficients[references, np.arange(2)]))
    weight = np.abs(display)**2
    sizes = 95 * weight / weight.max()
    # Draw physical bonds without joining the two ends of the finite device.
    finite_atoms = atoms.copy()
    finite_atoms.pbc = False
    i, j = neighbor_list('ij', finite_atoms, 1.6)
    bonds = np.stack((xyz[i[i < j]], xyz[j[i < j]]), axis=1)
    fig = plt.figure(figsize=(12, 8), layout='constrained')
    norm = Normalize(-np.pi, np.pi)
    for channel in range(2):
        phase = np.angle(display[:, channel])
        ax = fig.add_subplot(2, 2, channel + 1, projection='3d')
        ax.add_collection3d(Line3DCollection(bonds[:, :, [2, 0, 1]], colors='0.7', linewidths=.5))
        dots = ax.scatter(xyz[:, 2], xyz[:, 0], xyz[:, 1], s=sizes[:, channel],
                          c=phase, cmap='twilight', norm=norm, depthshade=False)
        ax.set(xlabel=r'$z$ ($\AA$)', ylabel=r'$x$ ($\AA$)', zlabel=r'$y$ ($\AA$)',
               title=f'Channel {channel + 1}: tau = {float(channels["tau_selected"][channel]):.8f}')
        ax.set_box_aspect((max(np.ptp(xyz[:, 2]), 1), np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1])))
        ax.view_init(elev=22, azim=-65)
        ax = fig.add_subplot(2, 2, channel + 3)
        ax.scatter(xyz[:, 2], theta / np.pi, s=sizes[:, channel], c=phase,
                   cmap='twilight', norm=norm)
        ax.set(xlabel=r'$z$ ($\AA$)', ylabel=r'Azimuth $\theta/\pi$',
               ylim=(-.08, 2.08), title='Unwrapped cylindrical surface')
        ax.grid(alpha=.2)
    colorbar = fig.colorbar(dots, ax=fig.axes, shrink=.65, pad=.03)
    colorbar.set_label('LCAO coefficient phase (rad)')
    colorbar.set_ticks([-np.pi, 0, np.pi], labels=[r'$-\pi$', '0', r'$\pi$'])
    energy = float(channels['energy'])
    fig.suptitle(f'(5,5) CNT, left incidence, E - E_F = {energy:g} eV\n'
                 'Marker area: |C|^2 (common scale); color: phase (display gauge)\n'
                 'Nonorthogonal LCAO weights; degenerate channels may rotate', fontsize=12)
    fig.savefig(output / 'eigenchannels.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks', type=int, default=10)
    parser.add_argument('--energy', type=float, default=0., help='Channel energy relative to E_F (eV)')
    parser.add_argument('--window', type=float, default=1., help='Scan from -window to +window (eV)')
    parser.add_argument('--points', type=int, default=101)
    parser.add_argument('--eta-lead', type=float, default=1e-5,
                        help='Lead broadening in eV; 1e-5 stabilizes the Dirac-point surface solve')
    parser.add_argument('--eta-device', type=float, default=1e-8,
                        help='Device broadening in eV; stabilizes the zero-energy block inversions')
    parser.add_argument('--output', type=Path, default=Path('cnt55_eigenchannels'))
    args = parser.parse_args()
    if args.blocks < 1:
        parser.error('--blocks must be positive')
    if not np.isfinite(args.window) or args.window <= 0 or args.points < 2:
        parser.error('--window must be finite and positive; --points must be at least 2')
    if not np.isfinite(args.energy):
        parser.error('--energy must be finite')
    if not np.isfinite(args.eta_lead) or args.eta_lead <= 0:
        parser.error('--eta-lead must be finite and positive')
    if not np.isfinite(args.eta_device) or args.eta_device < 0:
        parser.error('--eta-device must be finite and nonnegative')
    torch.set_num_threads(1)
    model = cnt_model()
    crossings = count_band_crossings(model, args.energy)
    if crossings != 2:
        parser.error(f'Channel energy has {crossings} positive band crossings; expected two')
    c, atoms = context_from_model(model, args.blocks, args.energy, args.eta_lead, args.eta_device)
    start = time.perf_counter()
    full_rgf(c)
    full_seconds = time.perf_counter() - start
    start = time.perf_counter()
    values = transmission_eigenvalues(c)
    values_seconds = time.perf_counter() - start
    start = time.perf_counter()
    channels = transmission_eigenchannels(c, n_channels=2, return_format='original')
    channel_seconds = time.perf_counter() - start
    # A finite device eta adds absorption proportional to the tube length.
    tolerance = 2e-6 + 10 * args.blocks * args.eta_device / 2.7
    np.testing.assert_allclose(values['tau_all'][:2].numpy(), [1., 1.], atol=tolerance, rtol=0)
    np.testing.assert_allclose(values['T_spectral'].numpy(), 2., atol=2*tolerance, rtol=0)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({'eigenvalues': values, 'eigenchannels': channels}, args.output / 'trans_eigs.pth')
    write(args.output / 'device.xyz', atoms)
    energies = np.linspace(-args.window, args.window, args.points)
    print(f'Scanning {args.points} energies in [-{args.window}, {args.window}] eV ...', flush=True)
    spectrum = scan_spectrum(model, args.blocks, energies, args.eta_lead, args.eta_device)
    np.savetxt(args.output / 'transmission_spectrum.csv', spectrum, delimiter=',',
               header='energy_minus_EF_eV,tau_1,tau_2,T_spectral,T_caroli_raw', comments='')
    plot_results(args.output, spectrum, atoms, channels)
    print('tau =', values['tau_all'].tolist())
    print('T =', float(values['T_spectral']))
    print(f'full RGF={full_seconds:.6f}s; eigenvalues={values_seconds:.6f}s; two channels={channel_seconds:.6f}s')
    print('coefficient bytes =', channels['coefficients'].numel() * channels['coefficients'].element_size())
    print('process peak RSS (KiB) =', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print('solve residual =', float(channels['diagnostics']['solve_relative_residual']))
    print('saved', args.output / 'trans_eigs.pth')
    print('spectrum:', args.output / 'transmission_spectrum.csv')
    print('plots:', args.output / 'transmission_eigenvalues.png', args.output / 'eigenchannels.png')


if __name__ == '__main__':
    main()
