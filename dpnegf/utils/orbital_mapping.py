"""Provenance and validation for device H/S orbital order."""
import copy
import json

import numpy as np
import torch


def write_orbital_mapping(cache, metadata):
    cache.create_dataset("orbital_mapping", data=json.dumps(metadata))


def read_orbital_mapping(cache):
    if "orbital_mapping" not in cache:
        return None
    value = cache["orbital_mapping"]
    if hasattr(value, "shape") and not isinstance(value, (np.ndarray, torch.Tensor)):
        value = value[()]
    if isinstance(value, (str, bytes)):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("Invalid H/S orbital_mapping metadata")
    return copy.deepcopy(value)


def _integers(values, size, name, permutation=False):
    a = np.asarray(values)
    if a.shape != (size,) or a.dtype.kind not in "iu":
        raise ValueError(f"Invalid {name}: expected {size} integer entries")
    if permutation and not np.array_equal(np.sort(a), np.arange(size)):
        raise ValueError(f"Invalid {name}: not a permutation")
    return a


def validate_orbital_mapping(metadata, expected, blocks, source):
    """Return an independent snapshot; never infer a missing cache permutation."""
    n = sum(blocks)
    if expected is not None and sum(expected["atom_norbs"]) != n:
        raise ValueError("Actual H/S dimensions conflict with current input orbital count")
    if metadata is None:
        return {"status": "unknown", "source": source, "block_sizes": list(blocks),
                "orbital_map": None, "inverse_orbital_map": None}
    if expected is None:
        raise ValueError("Cannot validate cached orbital mapping against current input")
    if metadata.get("version") != 1:
        raise ValueError("Unsupported H/S orbital mapping version")
    forward = _integers(metadata.get("orbital_map"), n, "orbital_map", True)
    inverse = _integers(metadata.get("inverse_orbital_map"), n, "inverse_orbital_map", True)
    if not np.array_equal(inverse[forward], np.arange(n)):
        raise ValueError("Cached orbital forward/inverse mappings disagree")
    natoms = len(expected["atomic_numbers"])
    atoms = _integers(metadata.get("atom_map"), natoms, "atom_map", True)
    norbs = _integers(metadata.get("atom_norbs"), natoms, "atom_norbs")
    if np.any(norbs <= 0) or sum(norbs) != n:
        raise ValueError("Cached atom orbital counts disagree with actual H/S dimensions")
    for key in ("atomic_numbers", "atom_norbs", "atom_map", "basis", "device_id",
                "orbital_map", "inverse_orbital_map"):
        if metadata.get(key) != expected[key]:
            raise ValueError(f"Cached orbital mapping conflicts with current input: {key}")
    positions = np.asarray(metadata.get("positions"))
    if (positions.shape != (natoms, 3) or not np.isfinite(positions).all()
            or not np.allclose(positions, expected["positions"], atol=1e-8, rtol=0)):
        raise ValueError("Cached atom order/positions conflict with current input")
    # Verify the orbital permutation also follows the declared per-atom order.
    offsets = np.concatenate(([0], np.cumsum(norbs)))
    by_atom = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in atoms])
    if not np.array_equal(forward, by_atom):
        raise ValueError("Cached orbital map disagrees with atom order/basis")
    result = copy.deepcopy(metadata)
    result.update(status="known", source=source, block_sizes=list(blocks))
    return result


def validate_hs_layout(parts, subblocks):
    """Check actual loaded H/S blocks against cache block dimensions."""
    hd, sd, hl, su, sl, hu = parts
    blocks = [int(x.shape[0]) for x in hd]
    declared = _integers(subblocks, len(blocks), "subblocks")
    if np.any(declared <= 0) or not np.array_equal(declared, blocks):
        raise ValueError("H/S cache subblocks disagree with actual H/S blocks")
    if len(sd) != len(blocks) or any(
            h.shape != (n, n) or s.shape != (n, n) for h, s, n in zip(hd, sd, blocks)):
        raise ValueError("H/S diagonal block dimensions disagree")
    for values, upper in ((hu, True), (su, True), (hl, False), (sl, False)):
        if len(values) != len(blocks) - 1 or any(
                b.shape != ((blocks[i], blocks[i + 1]) if upper else (blocks[i + 1], blocks[i]))
                for i, b in enumerate(values)):
            raise ValueError("H/S off-diagonal block dimensions disagree")
    return blocks
