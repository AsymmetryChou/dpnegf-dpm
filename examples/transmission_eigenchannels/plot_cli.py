"""Plot the standard dpnegf CLI output for this single-k CNT example."""
import argparse
from pathlib import Path

import numpy as np
import torch

from examples.transmission_eigenchannels.run_cnt55 import plot_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='Directory passed to dpnegf run -o')
    parser.add_argument('--energy', type=float, default=0., help='Requested channel energy (eV)')
    args = parser.parse_args()
    results = args.output / 'results'
    out = torch.load(results / 'negf.out.pth', map_location='cpu', weights_only=False)
    if len(out['k']) != 1:
        parser.error('This example plot expects exactly one k point')
    k = str(out['k'][0])
    spectrum = []
    for record in out['trans_eigvals'][k]:
        tau = record['tau_all'].numpy()
        top = np.pad(tau[:2], (0, max(0, 2 - len(tau))))
        spectrum.append([float(record['energy']), *top, float(record['T_spectral']),
                         float(record['T_caroli_raw'])])
    spectrum = np.asarray(spectrum)
    entries = out['trans_eigchannel'][k]
    entry = min(entries, key=lambda item: abs(float(item['energy']) - args.energy))
    if not np.isclose(float(entry['energy']), args.energy, atol=1e-7, rtol=0):
        parser.error(f'Channel energy {args.energy} eV is absent from the saved grid')
    # Load only the selected energy; the scan's coefficients stay on disk.
    channels = torch.load(results / entry['path'], map_location='cpu', weights_only=False)
    if channels['basis_order'] != 'original' or channels['coefficients'].shape[1] != 2:
        parser.error('Plots require two channels in original orbital order')
    import json
    config = json.loads((args.output / 'run_config.json').read_text())
    start, stop = map(int, config['task_options']['stru_options']['device']['id'].split('-'))
    # Use the saved mapping geometry: it describes the actual loaded H/S cache.
    from ase import Atoms
    mapping = channels['orbital_mapping']
    atoms = Atoms(numbers=mapping['atomic_numbers'], positions=mapping['positions'])
    if len(atoms) != stop - start or channels['coefficients'].shape[0] != len(atoms):
        parser.error('This plot requires one orbital per device carbon atom')
    figures = args.output / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    np.savetxt(figures / 'transmission_spectrum.csv', spectrum, delimiter=',',
               header='energy_minus_EF_eV,tau_1,tau_2,T_spectral,T_caroli_raw', comments='')
    plot_results(figures, spectrum, atoms, channels)
    print('Saved plots to', figures)
    print('E =', float(channels['energy']), 'eV; tau =', channels['tau_selected'].tolist())


if __name__ == '__main__':
    main()
