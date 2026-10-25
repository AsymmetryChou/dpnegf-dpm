import h5py
import numpy as np
import pytest
import torch
from dptb.data import AtomicDataDict
from dptb.data.transforms import OrbitalMapper
from dptb.nn.hr2hk import HR2HK

from dpnegf.utils.direct_btd import LocalBlocks, _build_profiles, write_block_rows
from dpnegf.utils.direct_btd_layout import AssemblyPlan, index_dtype
from dpnegf.utils.hs_cache import read_btd_kpoint
from dpnegf.tests.test_direct_btd import dense_from_parts



def local_data(dtype):
    mapper = OrbitalMapper({"C": ["2s", "2p", "3d"], "H": ["1s"]}, method="e3tb")
    mapper.get_orbpair_maps()
    generator = torch.Generator().manual_seed(83)
    data = {
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([
            [mapper.chemical_symbol_to_type[symbol]]
            for symbol in ("C", "H", "C", "H")
        ]),
        AtomicDataDict.EDGE_INDEX_KEY: torch.tensor(
            [[0, 0, 1, 2, 3, 0, 2, 2], [1, 1, 0, 1, 2, 0, 2, 3]]
        ),
        AtomicDataDict.EDGE_CELL_SHIFT_KEY: torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, 0],
            ],
            dtype=dtype,
        ),
        AtomicDataDict.KPOINT_KEY: torch.tensor(
            [[0, 0, 0], [0.17, -0.23, 0]], dtype=dtype
        ),
    }
    for key, count in (
        (AtomicDataDict.NODE_FEATURES_KEY, 4),
        (AtomicDataDict.NODE_OVERLAP_KEY, 4),
        (AtomicDataDict.EDGE_FEATURES_KEY, 8),
        (AtomicDataDict.EDGE_OVERLAP_KEY, 8),
    ):
        data[key] = torch.randn(
            count,
            mapper.reduced_matrix_element,
            generator=generator,
            dtype=dtype,
        )
    return mapper, data


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shared_pair_and_scatter_match_hr2hk(tmp_path, dtype):
    mapper, data = local_data(dtype)
    local_h = LocalBlocks(data, mapper)
    local_s = LocalBlocks(
        data, mapper, overlap=True, topology=local_h.topology
    )
    assert local_s.offsets is local_h.offsets
    assert local_s.masks is local_h.masks
    assert local_s.edge_orders is local_h.edge_orders
    blocks = [7, 13]
    tolerance = 2e-6 if dtype == torch.float32 else 1e-13
    with torch.no_grad():
        pair_plan = local_h.topology.cpu(mapper.full_basis_norb)
        assert local_s.topology.cpu(mapper.full_basis_norb) is pair_plan
        plan = AssemblyPlan(pair_plan, (0, 20), blocks, {})
        with h5py.File(tmp_path / "hs.h5", "w") as cache:
            cache["kpoints"] = data[AtomicDataDict.KPOINT_KEY].numpy()
            cache["subblocks"] = blocks
            for name in ("hd", "hu", "hl", "sd", "su", "sl"):
                cache.create_group(name)
            for k_index, kpoint in enumerate(data[AtomicDataDict.KPOINT_KEY]):
                phases = local_h.phases(kpoint)
                for prefix, local in (("h", local_h), ("s", local_s)):
                    write_block_rows(
                        cache,
                        prefix,
                        local,
                        kpoint,
                        k_index,
                        (0, 20),
                        blocks,
                        {},
                        phases=phases,
                        plan=plan,
                    )
        for overlap, indices in ((False, (0, 5, 2)), (True, (1, 3, 4))):
            reference = HR2HK(
                idp=mapper,
                dtype=dtype,
                overlap=overlap,
                edge_field=(
                    AtomicDataDict.EDGE_OVERLAP_KEY
                    if overlap
                    else AtomicDataDict.EDGE_FEATURES_KEY
                ),
                node_field=(
                    AtomicDataDict.NODE_OVERLAP_KEY
                    if overlap
                    else AtomicDataDict.NODE_FEATURES_KEY
                ),
            )(dict(data))[AtomicDataDict.HAMILTONIAN_KEY]
            with h5py.File(tmp_path / "hs.h5") as cache:
                for k_index, kpoint in enumerate(
                    data[AtomicDataDict.KPOINT_KEY]
                ):
                    parts = read_btd_kpoint(cache, kpoint, 0, "cpu")
                    actual = dense_from_parts(
                        [parts[index] for index in indices]
                    )
                    torch.testing.assert_close(
                        actual,
                        reference[k_index].cdouble(),
                        atol=tolerance,
                        rtol=tolerance,
                    )

def test_profiles_share_phases_and_match_legacy(monkeypatch):
    mapper, data = local_data(torch.float64)
    local_h = LocalBlocks(data, mapper)
    local_s = LocalBlocks(
        data, mapper, overlap=True, topology=local_h.topology
    )
    kpoints = data[AtomicDataDict.KPOINT_KEY]
    windows = {"lead_L": (0, 3), "lead_R": (17, 20)}
    independent_s = LocalBlocks(data, mapper, overlap=True)
    with torch.no_grad():
        expected = _build_profiles(
            (local_h, independent_s),
            kpoints,
            local_h.offsets,
            (3, 17),
            windows,
        )
    count = []
    original = local_h.phases

    def phases(kpoint):
        count.append(1)
        return original(kpoint)

    monkeypatch.setattr(local_h, "phases", phases)
    with torch.no_grad():
        actual = _build_profiles(
            (local_h, local_s),
            kpoints,
            local_h.offsets,
            (3, 17),
            windows,
        )
    assert len(count) == len(kpoints)
    for index in (0, 1):
        np.testing.assert_array_equal(actual[index], expected[index])
    assert actual[2] == expected[2]

def test_cached_masks_are_not_reconverted(monkeypatch):
    mapper, data = local_data(torch.float64)
    local = LocalBlocks(data, mapper)
    original = torch.Tensor.numpy

    def guarded_numpy(tensor, *args, **kwargs):
        assert tensor.dtype != torch.bool
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "numpy", guarded_numpy)
    with torch.no_grad():
        for source in range(4):
            list(local.pairs([0, 0, 0], atoms=[source]))
        local.topology.cpu(mapper.full_basis_norb)

@pytest.mark.parametrize("failure", ["non_neighbour", "contact"])
def test_plan_rejects_lost_connections(tmp_path, failure):
    mapper, data = local_data(torch.float64)
    local = LocalBlocks(data, mapper)
    with torch.no_grad(), h5py.File(tmp_path / "invalid.h5", "w") as cache:
        for name in ("hd", "hu", "hl"):
            cache.create_group(name)
        device = (0, 20) if failure == "non_neighbour" else (0, 19)
        blocks = [2] * 10 if failure == "non_neighbour" else [10, 9]
        windows = {} if failure == "non_neighbour" else {"lead_L": (19, 20)}
        plan = AssemblyPlan(
            local.topology.cpu(mapper.full_basis_norb),
            device,
            blocks,
            windows,
        )
        with pytest.raises(ValueError, match="Non-neighbouring|boundary"):
            write_block_rows(
                cache,
                "h",
                local,
                [0, 0, 0],
                0,
                device,
                blocks,
                windows,
                plan=plan,
            )


def test_overlap_only_non_gamma_profile():
    mapper = OrbitalMapper({"H": ["1s"]}, method="e3tb")
    data = {
        AtomicDataDict.ATOM_TYPE_KEY: torch.zeros((4, 1), dtype=torch.long),
        AtomicDataDict.EDGE_INDEX_KEY: torch.tensor([[0, 0], [3, 3]]),
        AtomicDataDict.EDGE_CELL_SHIFT_KEY: torch.tensor(
            [[0.0, 0, 0], [1.0, 0, 0]], dtype=torch.float64
        ),
        AtomicDataDict.NODE_FEATURES_KEY: torch.zeros(
            (4, 1), dtype=torch.float64
        ),
        AtomicDataDict.EDGE_FEATURES_KEY: torch.zeros(
            (2, 1), dtype=torch.float64
        ),
        AtomicDataDict.NODE_OVERLAP_KEY: torch.ones(
            (4, 1), dtype=torch.float64
        ),
        AtomicDataDict.EDGE_OVERLAP_KEY: torch.tensor(
            [[1.0], [-1.0]], dtype=torch.float64
        ),
    }
    local_h = LocalBlocks(data, mapper)
    local_s = LocalBlocks(
        data, mapper, overlap=True, topology=local_h.topology
    )
    with torch.no_grad():
        gamma = _build_profiles(
            (local_h, local_s), [[0, 0, 0]], local_h.offsets, (0, 4), {}
        )
        union = _build_profiles(
            (local_h, local_s),
            [[0, 0, 0], [0.25, 0, 0]],
            local_h.offsets,
            (0, 4),
            {},
        )
    assert gamma[0][0] == 1
    assert union[0][0] == 4


@pytest.mark.parametrize("orthogonal", [False, True])
def test_planned_contacts_and_orthogonal_match_streaming(tmp_path, orthogonal):
    mapper, data = local_data(torch.float64)
    local = LocalBlocks(data, mapper)
    device = (3, 17)
    blocks = [7, 7]
    windows = {"lead_L": (0, 3), "lead_R": (17, 20)}
    with torch.no_grad():
        plan = AssemblyPlan(
            local.topology.cpu(mapper.full_basis_norb),
            device,
            blocks,
            windows,
        )
        matrices = []
        for name, assembly in (("legacy", None), ("planned", plan)):
            with h5py.File(tmp_path / f"{name}.h5", "w") as cache:
                for group in ("hd", "hu", "hl"):
                    cache.create_group(group)
                matrices.append(
                    write_block_rows(
                        cache,
                        "h",
                        local,
                        [0.17, -0.23, 0],
                        0,
                        device,
                        blocks,
                        windows,
                        factor=2.5,
                        orthogonal=orthogonal,
                        plan=assembly,
                    )
                )
        for expected, actual in zip(*matrices):
            for tab in windows:
                torch.testing.assert_close(
                    actual[tab], expected[tab], atol=1e-13, rtol=1e-13
                )
        with h5py.File(tmp_path / "legacy.h5") as expected, h5py.File(
            tmp_path / "planned.h5"
        ) as actual:
            for group in ("hd", "hu", "hl"):
                assert set(expected[group]) == set(actual[group])
                for dataset in expected[group]:
                    np.testing.assert_allclose(
                        actual[group][dataset][()],
                        expected[group][dataset][()],
                        atol=1e-13,
                        rtol=1e-13,
                    )

def test_pair_plan_empty_edges():
    mapper = OrbitalMapper({"H": ["1s"]}, method="e3tb")
    data = {
        AtomicDataDict.ATOM_TYPE_KEY: torch.zeros((2, 1), dtype=torch.long),
        AtomicDataDict.EDGE_INDEX_KEY: torch.empty((2, 0), dtype=torch.long),
        AtomicDataDict.EDGE_CELL_SHIFT_KEY: torch.empty(
            (0, 3), dtype=torch.float64
        ),
        AtomicDataDict.NODE_FEATURES_KEY: torch.tensor(
            [[2.0], [4.0]], dtype=torch.float64
        ),
        AtomicDataDict.EDGE_FEATURES_KEY: torch.empty(
            (0, 1), dtype=torch.float64
        ),
    }
    with torch.no_grad():
        local = LocalBlocks(data, mapper)
        values = local.topology.cpu(1).evaluate(local, local.phases([0, 0, 0]))
    np.testing.assert_array_equal(values.ravel(), [2, 4])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_chunked_pairs_preserve_exact_values_and_inputs(monkeypatch, dtype):
    import dpnegf.utils.direct_btd_layout as layout

    mapper, data = local_data(dtype)
    monkeypatch.setattr(layout, "pair_chunk_size", lambda full_norb, itemsize: 2)
    with torch.no_grad():
        local = LocalBlocks(data, mapper)
        plan = local.topology.cpu(mapper.full_basis_norb)
        original_nodes = local.nodes.clone()
        original_hoppings = local.hoppings.clone()
        for kpoint in data[AtomicDataDict.KPOINT_KEY]:
            phases = local.phases(kpoint)
            raw = np.zeros(
                (len(plan.sources), plan.full_norb, plan.full_norb),
                dtype=phases.numpy().dtype,
            )
            raw[plan.node_pairs] = local.nodes.numpy()
            np.add.at(
                raw,
                plan.edge_pairs,
                local.hoppings.numpy() * phases.numpy()[:, None, None],
            )
            expected = raw + raw[plan.reverse].transpose(0, 2, 1).conj()
            actual = plan.evaluate(local, phases)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(
                plan.nonzero(actual), expected.ravel()[plan.value_indices] != 0
            )
        torch.testing.assert_close(local.nodes, original_nodes, rtol=0, atol=0)
        torch.testing.assert_close(
            local.hoppings, original_hoppings, rtol=0, atol=0
        )
    masks = np.asarray(local.topology.cpu_masks)
    valid = masks[plan.sources, :, None] & masks[plan.targets, None, :]
    np.testing.assert_array_equal(plan.value_indices, np.flatnonzero(valid))
    for name in (
        "sources",
        "targets",
        "node_pairs",
        "edge_pairs",
        "reverse",
        "starts",
        "rows",
        "columns",
        "value_indices",
    ):
        assert getattr(plan, name).dtype == np.int32
    assembly = AssemblyPlan(plan, (0, 20), [7, 13], {})
    for operations in assembly.operations:
        for column, source, destination in operations:
            assert source.dtype == destination.dtype == np.int32


@pytest.mark.parametrize(
    "maximum,expected",
    [
        (0, np.int32),
        (2**31 - 1, np.int32),
        (2**31, np.int64),
        (2**40, np.int64),
    ],
)
def test_index_dtype_capacity(maximum, expected):
    assert index_dtype(maximum) == expected

def test_large_window_destination_does_not_overflow():
    from dpnegf.utils.direct_btd_layout import window_indices

    selected, destination = window_indices(
        np.array([49999], dtype=np.int32), np.array([49999], dtype=np.int32),
        (0, 50000), (0, 50000),
    )
    np.testing.assert_array_equal(selected, [0])
    np.testing.assert_array_equal(destination, [2499999999])
    assert destination.dtype == np.int64

def test_small_window_at_large_orbital_offset():
    from dpnegf.utils.direct_btd_layout import window_indices

    offset = 2**32
    selected, destination = window_indices(
        np.array([offset + 1], dtype=np.int64),
        np.array([offset + 1], dtype=np.int64),
        (offset, offset + 2),
        (offset, offset + 2),
    )
    np.testing.assert_array_equal(selected, [0])
    np.testing.assert_array_equal(destination, [3])
    assert destination.dtype == np.int32


def test_pair_keys_remain_int64_with_compact_atom_indices():
    from types import SimpleNamespace
    from dpnegf.utils.direct_btd_layout import PairPlan

    atom_count = 50000
    mask = np.array([True])
    topology = SimpleNamespace(
        masks=[mask] * atom_count,
        cpu_masks=[mask] * atom_count,
        edges=np.array([[0, 49999], [49999, 0]], dtype=np.int64),
        offsets=np.arange(atom_count + 1, dtype=np.int64),
    )
    plan = PairPlan(topology, 1)
    np.testing.assert_array_equal(plan.sources[plan.reverse], plan.targets)
    np.testing.assert_array_equal(plan.targets[plan.reverse], plan.sources)
    np.testing.assert_array_equal(
        plan.sources[plan.node_pairs], np.arange(atom_count)
    )


def test_nonzero_scan_crosses_temporary_chunk_boundary():
    from types import SimpleNamespace
    from dpnegf.utils.direct_btd_layout import PairPlan

    values = np.zeros(300003, dtype=np.complex128)
    values[[0, 262143, 262144, -1]] = [1, 2j, -3, 4j]
    indices = np.arange(len(values) - 1, -1, -1, dtype=np.int32)
    plan = SimpleNamespace(value_indices=indices)
    np.testing.assert_array_equal(
        PairPlan.nonzero(plan, values), values[indices] != 0
    )