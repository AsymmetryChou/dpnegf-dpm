"""Cache provenance, corruption checks and unknown legacy orbital ordering."""
import copy
from dataclasses import replace

import h5py
import numpy as np
import pytest
import torch

from dpnegf.negf.negf_hamiltonian_init import NEGFHamiltonianInit
from dpnegf.utils.orbital_mapping import validate_orbital_mapping, write_orbital_mapping


def metadata():
    return dict(version=1, orbital_map=[1, 2, 0], inverse_orbital_map=[2, 0, 1],
                atom_map=[1, 0], atom_norbs=[1, 2], atomic_numbers=[1, 6],
                positions=[[1., 0., 0.], [0., 0., 0.]], device_id=[2, 4],
                basis={'convention': 'test', 'basis': {'H': ['s'], 'C': ['s', 'p']}})


def cache(tmp_path, fmt, btd, mapping):
    data = dict(kpoints=[[0., 0., 0.]], subblocks=[2, 1] if btd else [3])
    if btd:
        data.update(hd=[[torch.eye(2).cdouble(), torch.eye(1).cdouble()]],
                    sd=[[torch.eye(2).cdouble(), torch.eye(1).cdouble()]])
        for key in ('hu', 'su', 'hl', 'sl'):
            data[key] = [[torch.zeros((2, 1) if key in ('hu', 'su') else (1, 2), dtype=torch.complex128)]]
    else:
        data.update(HD=torch.eye(3).cdouble()[None, None], SD=torch.eye(3).cdouble()[None, None])
    path = tmp_path / ('HS_device.' + fmt)
    if fmt == 'pth':
        if mapping is not None:
            data['orbital_mapping'] = mapping
        torch.save(data, path)
    else:
        with h5py.File(path, 'w') as f:
            if mapping is not None:
                write_orbital_mapping(f, mapping)
            for key, value in data.items():
                if key in ('hd', 'sd', 'hu', 'hl', 'su', 'sl'):
                    group = f.create_group(key)
                    for i, block in enumerate(value[0]):
                        group.create_dataset(f'{key}_k0_b{i}_real', data=block.real.numpy())
                        group.create_dataset(f'{key}_k0_b{i}_imag', data=block.imag.numpy())
                else:
                    f.create_dataset(key, data=np.asarray(value))
    return path


def owner(tmp_path):
    ham = NEGFHamiltonianInit.__new__(NEGFHamiltonianInit)
    ham.saved_HS_path = str(tmp_path)
    ham.torch_device = torch.device('cpu')
    ham._input_orbital_mapping = metadata()
    return ham


@pytest.mark.parametrize('fmt', ['h5', 'pth'])
@pytest.mark.parametrize('btd', [False, True])
@pytest.mark.parametrize('known', [False, True])
def test_loaded_cache_mapping(tmp_path, fmt, btd, known):
    path = cache(tmp_path, fmt, btd, metadata() if known else None)
    ham = owner(tmp_path)
    # Input-derived convenience attributes must never supply cache provenance.
    ham.device_orbital_map = [0, 1, 2]
    ham.get_hs_device([0., 0., 0.], 0., btd)
    snapshot = ham.get_orbital_map()
    assert snapshot['status'] == ('known' if known else 'unknown')
    assert snapshot['source'] == str(path)
    assert snapshot['block_sizes'] == ([2, 1] if btd else [3])
    assert snapshot['orbital_map'] == ([1, 2, 0] if known else None)
    snapshot['block_sizes'][0] = 100
    assert ham.get_orbital_map()['block_sizes'][0] != 100


@pytest.mark.parametrize('key,value', [
    ('orbital_map', [1, 1, 0]), ('orbital_map', [1., 2., 0.]),
    ('inverse_orbital_map', [0, 1, 2]), ('atom_map', [0, 1]),
    ('atom_norbs', [2, 1]), ('atomic_numbers', [6, 1]),
    ('positions', [[0., 0., 0.], [1., 0., 0.]]),
    ('basis', {'convention': 'other'}), ('device_id', [0, 2]),
])
@pytest.mark.parametrize('fmt', ['h5', 'pth'])
def test_conflicting_mapping_fails_on_load(tmp_path, fmt, key, value):
    bad = metadata()
    bad[key] = value
    cache(tmp_path, fmt, True, bad)
    with pytest.raises(ValueError):
        owner(tmp_path).get_hs_device([0., 0., 0.], 0., True)


def test_h5_precedence_and_failed_load_invalidates_mapping(tmp_path):
    cache(tmp_path, 'pth', False, metadata())
    path = cache(tmp_path, 'h5', False, None)
    ham = owner(tmp_path)
    ham.get_hs_device([0., 0., 0.], 0., False)
    assert ham.get_orbital_map()['status'] == 'unknown'
    assert ham.get_orbital_map()['source'] == str(path)
    with pytest.raises(AssertionError):
        ham.get_hs_device([1., 0., 0.], 0., False)
    with pytest.raises(RuntimeError):
        ham.get_orbital_map()


@pytest.mark.parametrize('mapping', [None, metadata()])
def test_actual_dimensions_and_block_sizes(tmp_path, mapping):
    path = cache(tmp_path, 'pth', True, mapping)
    data = torch.load(path, weights_only=False)
    data['subblocks'] = [1, 2]
    torch.save(data, path)
    with pytest.raises(ValueError, match='subblocks'):
        owner(tmp_path).get_hs_device([0., 0., 0.], 0., True)
    with pytest.raises(ValueError, match='dimensions'):
        validate_orbital_mapping(mapping, metadata(), [4], 'test')


def test_unknown_channels_and_device_mapping_snapshot(tmp_path):
    from .test_trans_eigs import _model
    from .test_device_property_batched import _make_deviceprop
    from dpnegf.negf.trans_eigs import transmission_eigenvalues, transmission_eigenchannels
    c, *_ = _model()
    unknown = replace(c, orbital_map=None, orbital_map_known=False)
    assert transmission_eigenvalues(unknown)['orbital_map'] is None
    assert transmission_eigenchannels(unknown)['orbital_mapping_status'] == 'unknown'
    with pytest.raises(ValueError, match='unknown'):
        transmission_eigenchannels(unknown, return_format='original')
    dev = _make_deviceprop([3, 4, 2], c.hd, c.sd, c.hl, c.hu, c.sl, c.su)
    mapping = dict(status='known', orbital_map=list(range(8, -1, -1)),
                   inverse_orbital_map=list(range(8, -1, -1)), block_sizes=[3, 4, 2])
    dev.hamiltonian.get_orbital_map = lambda: copy.deepcopy(mapping)
    dev.lead_L.se, dev.lead_R.se = c.left_se, c.right_se
    dev.cal_green_function(torch.tensor([.17]), [0, 0, 0], need_trans_eigs=True)
    mapping['orbital_map'] = list(range(9))
    dev.hamiltonian.device_orbital_map = list(range(9))
    result = next(dev.iter_transmission_eigs(channels=True, return_format='original'))
    assert result['orbital_map'].tolist() == list(range(8, -1, -1))
    assert result['orbital_mapping']['orbital_map'] == list(range(8, -1, -1))


@pytest.mark.parametrize('key', ['sd', 'hu', 'sl'])
def test_actual_hs_shapes_are_validated(tmp_path, key):
    path = cache(tmp_path, 'pth', True, metadata())
    data = torch.load(path, weights_only=False)
    data[key][0][0] = torch.zeros(4, 4, dtype=torch.complex128)
    torch.save(data, path)
    with pytest.raises(ValueError, match='dimensions'):
        owner(tmp_path).get_hs_device([0., 0., 0.], 0., True)


def test_legacy_pth_without_declared_blocks(tmp_path):
    path = cache(tmp_path, 'pth', True, None)
    data = torch.load(path, weights_only=False)
    del data['subblocks']
    torch.save(data, path)
    ham = owner(tmp_path)
    ham.get_hs_device([0., 0., 0.], 0., True)
    assert ham.get_orbital_map()['status'] == 'unknown'
    assert ham.get_orbital_map()['block_sizes'] == [2, 1]
