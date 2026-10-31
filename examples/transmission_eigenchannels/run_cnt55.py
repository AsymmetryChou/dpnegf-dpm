"""Perfect (5,5) CNT: nearest-neighbor pi model, one orbital per carbon.

Run from the repository root:
    python -m examples.transmission_eigenchannels.run_cnt55 --blocks 100

Builds the actual ASE CNT geometry and its periodic H/S blocks, uses DPNEGF
surface self-energies and full RGF, and writes two complex device channels.
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


def cnt_context(nblocks=10, energy=.2, overlap=.1):
    """Identical periodic device and leads; hopping -2.7 eV, nearest neighbors."""
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
    crossings = int(((bands[:-1] < energy) & (bands[1:] >= energy)).sum())
    if crossings != 2:
        raise ValueError(f'Chosen energy has {crossings} positive band crossings; expected two')
    # surface_green(h01) is the surface extending toward the positive index.
    right, _ = selfEnergy(h0, h1, s0, s1, energy, hDL=h1, sDL=s1,
                          etaLead=1e-8, numba_jit=False)
    left, _ = selfEnergy(h0, h1.mH.resolve_conj(), s0, s1.mH.resolve_conj(), energy,
                         hDL=h1.mH.resolve_conj(), sDL=s1.mH.resolve_conj(),
                         etaLead=1e-8, numba_jit=False)
    context = TransportContext(energy, [h0] * nblocks, [s0] * nblocks,
                               [h1] * (nblocks - 1), [h1.mH] * (nblocks - 1),
                               [s1] * (nblocks - 1), [s1.mH] * (nblocks - 1),
                               left, right, eta_device=0., eta_lead=1e-8, kpoint=[0., 0., 0.])
    return context, unit.repeat((1, 1, nblocks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks', type=int, default=10)
    parser.add_argument('--energy', type=float, default=.2)
    parser.add_argument('--output', type=Path, default=Path('cnt55_eigenchannels'))
    args = parser.parse_args()
    if args.blocks < 1:
        parser.error('--blocks must be positive')
    torch.set_num_threads(1)
    c, atoms = cnt_context(args.blocks, args.energy)
    start = time.perf_counter()
    full = recursive_gf(c.energy, c.hl, c.hd, c.hu, c.sd, c.su, c.sl,
                        c.left_se, c.right_se, eta=c.eta_device, keep_gr_left=True)
    c.gr_left = full[5]
    full_seconds = time.perf_counter() - start
    start = time.perf_counter()
    values = transmission_eigenvalues(c)
    values_seconds = time.perf_counter() - start
    start = time.perf_counter()
    channels = transmission_eigenchannels(c, n_channels=2, return_format='original')
    channel_seconds = time.perf_counter() - start
    np.testing.assert_allclose(values['tau_all'][:2].numpy(), [1., 1.], atol=2e-6)
    np.testing.assert_allclose(values['T_spectral'].numpy(), 2., atol=4e-6)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({'eigenvalues': values, 'eigenchannels': channels}, args.output / 'trans_eigs.pth')
    write(args.output / 'device.xyz', atoms)
    print('tau =', values['tau_all'].tolist())
    print('T =', float(values['T_spectral']))
    print(f'full RGF={full_seconds:.6f}s; eigenvalues={values_seconds:.6f}s; two channels={channel_seconds:.6f}s')
    print('coefficient bytes =', channels['coefficients'].numel() * channels['coefficients'].element_size())
    print('process peak RSS (KiB) =', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print('solve residual =', float(channels['diagnostics']['solve_relative_residual']))
    print('saved', args.output / 'trans_eigs.pth')


if __name__ == '__main__':
    main()
