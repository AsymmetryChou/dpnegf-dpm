import numpy as np
import torch 
from dptb.data import AtomicDataDict


def index_dtype(maximum):
    return np.int32 if maximum <= np.iinfo(np.int32).max else np.int64

def pair_chunk_size(full_norb, itemsize):
    return max(1, min(16384, 4 * 1024**2 // (full_norb**2 * itemsize)))


class LocalTopology:
    """Share atom masks, directed-pair indices and local orbital coordinates."""

    def __init__(self, data, idp):
        atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].flatten().cpu().tolist()
        self.masks = [idp.mask_to_basis[atom_type] for atom_type in atom_types]
        type_indices = {
            atom_type: np.flatnonzero(idp.mask_to_basis[atom_type].cpu().numpy())
            for atom_type in set(atom_types)
        }
        self.indices = [type_indices[atom_type] for atom_type in atom_types]
        self.offsets = np.cumsum([0] + [len(indices) for indices in self.indices])
        self.edges = data[AtomicDataDict.EDGE_INDEX_KEY].cpu().numpy()
        self.shifts = data[AtomicDataDict.EDGE_CELL_SHIFT_KEY]
        self.edge_orders = [
            np.argsort(endpoints, kind="stable") for endpoints in self.edges
        ]
        self.edge_starts = [
            np.r_[0, np.cumsum(np.bincount(endpoints, minlength=len(atom_types)))]
            for endpoints in self.edges
        ]
        self.cpu_masks = (
            [mask.numpy() for mask in self.masks]
            if self.shifts.device.type == "cpu"
            else None
        )
        self.cpu_plan = None

    def cpu(self, full_norb):
        if self.cpu_plan is None:
            self.cpu_plan = PairPlan(self, full_norb)
        return self.cpu_plan


class PairPlan:
    """Index only local atom pairs, never the global orbital square."""

    def __init__(self, topology, full_norb):
        atom_count = len(topology.masks)
        atoms = np.arange(atom_count, dtype=np.int64)
        source, target = topology.edges
        keys = np.unique(np.concatenate((
            atoms * atom_count + atoms,
            source * atom_count + target,
            target * atom_count + source,
        )))
        atom_dtype = index_dtype(atom_count - 1)
        pair_count = len(keys)
        pair_dtype = index_dtype(pair_count - 1)
        self.sources = (keys // atom_count).astype(atom_dtype)
        self.targets = (keys % atom_count).astype(atom_dtype)
        self.node_pairs = np.searchsorted(keys, atoms * atom_count + atoms).astype(pair_dtype)
        self.edge_pairs = np.searchsorted(keys, source * atom_count + target).astype(pair_dtype)
        self.reverse = np.searchsorted(
            keys, self.targets.astype(np.int64) * atom_count + self.sources
        ).astype(pair_dtype)
        del keys
        self.starts = np.r_[0, np.cumsum(np.bincount(
            self.sources, minlength=atom_count
        ))].astype(index_dtype(pair_count))
        self.full_norb = full_norb
        self.topology = topology
        masks = np.asarray(topology.cpu_masks)
        orbital_counts = np.diff(topology.offsets)
        element_starts = np.r_[0, np.cumsum(
            orbital_counts[self.sources] * orbital_counts[self.targets]
        )]
        element_count = int(element_starts[-1])
        self.value_indices = np.empty(
            element_count, dtype=index_dtype(pair_count * full_norb**2 - 1)
        )
        coordinate_dtype = index_dtype(int(topology.offsets[-1]) - 1)
        self.rows = np.empty(element_count, dtype=coordinate_dtype)
        self.columns = np.empty(element_count, dtype=coordinate_dtype)
        orbital_rank = np.cumsum(masks, axis=1) - 1
        row_basis = np.arange(full_norb)[None, :, None]
        column_basis = np.arange(full_norb)[None, None, :]
        chunk_size = pair_chunk_size(full_norb, 8)
        for start in range(0, pair_count, chunk_size):
            stop = min(start + chunk_size, pair_count)
            sources = self.sources[start:stop]
            targets = self.targets[start:stop]
            valid = masks[sources, :, None] & masks[targets, None, :]
            indices = np.flatnonzero(valid)
            selection = slice(element_starts[start], element_starts[stop])
            self.value_indices[selection] = indices + start * full_norb**2
            rows = topology.offsets[sources, None, None] + orbital_rank[
                sources[:, None, None], row_basis
            ]
            columns = topology.offsets[targets, None, None] + orbital_rank[
                targets[:, None, None], column_basis
            ]
            self.rows[selection] = np.broadcast_to(rows, valid.shape).ravel()[indices]
            self.columns[selection] = np.broadcast_to(columns, valid.shape).ravel()[indices]

    def evaluate(self, local, phases):
        phases = phases.numpy()
        raw = np.zeros(
            (len(self.sources), self.full_norb, self.full_norb),
            dtype=phases.dtype,
        )
        raw[self.node_pairs] = local.nodes.numpy()
        chunk_size = pair_chunk_size(self.full_norb, raw.dtype.itemsize)
        hopping = local.hoppings.numpy()
        for start in range(0, len(hopping), chunk_size):
            stop = start + chunk_size
            np.add.at(
                raw, self.edge_pairs[start:stop],
                hopping[start:stop] * phases[start:stop, None, None],
            )
        values = np.empty_like(raw)
        for start in range(0, len(raw), chunk_size):
            stop = min(start + chunk_size, len(raw))
            output = values[start:stop]
            np.conjugate(raw[self.reverse[start:stop]].transpose(0, 2, 1), out=output)
            np.add(raw[start:stop], output, out=output)
        return values

    def nonzero(self, values):
        flat = values.ravel()
        result = np.empty(len(self.value_indices), dtype=bool)
        chunk_size = max(1, 4 * 1024**2 // values.dtype.itemsize)
        for start in range(0, len(result), chunk_size):
            stop = min(start + chunk_size, len(result))
            np.not_equal(flat[self.value_indices[start:stop]], 0, out=result[start:stop])
        return result
    

def window_indices(rows, columns, row_window, column_window):
    selected = np.flatnonzero(
        (rows >= row_window[0]) & (rows < row_window[1])
        & (columns >= column_window[0]) & (columns < column_window[1])
    ).astype(index_dtype(len(rows) - 1))
    destination_dtype = index_dtype(
        int(row_window[1] - row_window[0]) * int(column_window[1] - column_window[0]) - 1
    )
    destination = (
        (rows[selected] - row_window[0]).astype(destination_dtype) * (column_window[1] - column_window[0])
        + (columns[selected] - column_window[0]).astype(destination_dtype)
    )
    return selected, destination.astype(destination_dtype, copy=False)


class AssemblyPlan:
    """Precompute element destinations shared by H, S and every k point."""

    def __init__(self, pair_plan, device_window, blocks, lead_windows):
        self.pairs = pair_plan
        self.blocks = blocks
        self.bounds = np.cumsum([device_window[0]] + list(blocks))
        rows, columns = pair_plan.rows, pair_plan.columns
        slot_dtype = index_dtype(3 * len(blocks))
        row_blocks = (np.searchsorted(self.bounds, rows, side="right") - 1).astype(slot_dtype)
        column_blocks = (np.searchsorted(self.bounds, columns, side="right") - 1).astype(slot_dtype)
        inside = (
            (rows >= device_window[0]) & (rows < device_window[1])
            & (columns >= device_window[0]) & (columns < device_window[1])
        )
        near = np.abs(row_blocks - column_blocks) <= 1
        self.forbidden = pair_plan.value_indices[inside & ~near]
        selected = np.flatnonzero(inside & near).astype(index_dtype(len(rows) - 1))
        slots = row_blocks[selected] * 3 + (column_blocks[selected] - row_blocks[selected] + 1)
        order = np.argsort(slots, kind="stable")
        selected = selected[order]
        counts = np.bincount(slots, minlength=3 * len(blocks))
        starts = np.r_[0, np.cumsum(counts)]
        del order, slots, inside, near, row_blocks, column_blocks
        self.operations = []
        for row_block, size in enumerate(blocks):
            operations = []
            for column_block in range(max(0, row_block - 1), min(len(blocks), row_block + 2)):
                slot = row_block * 3 + column_block - row_block + 1
                indices = selected[starts[slot]:starts[slot + 1]]
                destination_dtype = index_dtype(int(size) * int(blocks[column_block]) - 1)
                destination = (
                    (rows[indices] - self.bounds[row_block]).astype(destination_dtype) * blocks[column_block]
                    + (columns[indices] - self.bounds[column_block]).astype(destination_dtype)
                )
                operations.append((
                    column_block, pair_plan.value_indices[indices],
                    destination.astype(destination_dtype, copy=False),
                ))
            self.operations.append(operations)
        del indices, selected, counts, starts
        self.contacts = {}
        self.principal = {}
        self.forbidden_contacts = []
        for tab, window in lead_windows.items():
            boundary = 0 if tab == "lead_L" else len(blocks) - 1
            boundary_window = self.bounds[boundary:boundary + 2]
            selected, destination = window_indices(rows, columns, boundary_window, window)
            shape = (blocks[boundary], window[1] - window[0])
            self.contacts[tab] = (shape, pair_plan.value_indices[selected], destination)
            selected, destination = window_indices(rows, columns, window, window)
            self.principal[tab] = (
                (window[1] - window[0],) * 2, pair_plan.value_indices[selected], destination
            )
            invalid = (
                (rows >= device_window[0]) & (rows < device_window[1])
                & (columns >= window[0]) & (columns < window[1])
                & ((rows < boundary_window[0]) | (rows >= boundary_window[1]))
            )
            self.forbidden_contacts.append(pair_plan.value_indices[invalid])

    def write(self, cache, prefix, values, k_index, factor, orthogonal, write_complex):
        flat = None if orthogonal else values.ravel()
        if not orthogonal:
            if np.any(flat[self.forbidden] != 0):
                raise ValueError("Non-neighbouring block contribution would be lost.")
            if any(np.any(flat[indices] != 0) for indices in self.forbidden_contacts):
                raise ValueError("Device-lead coupling extends beyond its boundary block.")
        dtype = values.dtype
        workspace = np.empty((3, max(self.blocks) ** 2), dtype=dtype)
        for row_block, operations in enumerate(self.operations):
            for slot, (column_block, source, destination) in enumerate(operations):
                shape = (self.blocks[row_block], self.blocks[column_block])
                output = workspace[slot, :shape[0] * shape[1]].reshape(shape)
                output.fill(0)
                if orthogonal:
                    if row_block == column_block:
                        np.fill_diagonal(output, 1)
                else:
                    output.ravel()[destination] = flat[source]
                output *= factor
                suffix = "d" if column_block == row_block else ("u" if column_block > row_block else "l")
                name = prefix + suffix
                write_complex(cache[name], f"{name}_k{k_index}_b{min(row_block, column_block)}",
                              torch.from_numpy(output))
        results = []
        for entries in (self.contacts, self.principal):
            matrices = {}
            for tab, (shape, source, destination) in entries.items():
                output = np.zeros(shape, dtype=dtype)
                if orthogonal:
                    if entries is self.principal:
                        np.fill_diagonal(output, 1)
                else:
                    output.ravel()[destination] = flat[source]
                matrices[tab] = torch.from_numpy(output)
            results.append(matrices)
        return tuple(results)


class CudaTopology:
    """GPU-resident pair CSR; only orbital offsets are mirrored on the host."""

    def __init__(self, data, idp):
        self.atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].flatten().long()
        self.device = self.atom_types.device
        if self.device.type != "cuda":
            raise ValueError("CudaTopology requires CUDA data.")
        self.type_masks = idp.mask_to_basis.to(self.device)
        self.type_ranks = self.type_masks.long().cumsum(1) - 1
        counts = self.type_masks.sum(1)[self.atom_types]
        self.offsets_gpu = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        self.offsets = self.offsets_gpu.cpu().numpy()
        self.edges = data[AtomicDataDict.EDGE_INDEX_KEY].long()
        self.shifts = data[AtomicDataDict.EDGE_CELL_SHIFT_KEY]
        self.full_norb = idp.full_basis_norb
        atom_count = len(self.atom_types)
        atoms = torch.arange(atom_count, device=self.device)
        source, target = self.edges
        forward_keys = source * atom_count + target
        keys = torch.unique(torch.cat((
            atoms * atom_count + atoms, forward_keys, target * atom_count + source,
        )), sorted=True)
        self.sources = keys // atom_count
        self.targets = keys % atom_count
        self.reverse = torch.searchsorted(keys, self.targets * atom_count + self.sources)
        edge_pairs = torch.searchsorted(keys, forward_keys)
        self.edge_order = torch.argsort(edge_pairs, stable=True)
        edge_counts = torch.bincount(edge_pairs, minlength=len(keys))
        self.edge_ptr = torch.cat((edge_counts.new_zeros(1), edge_counts.cumsum(0)))
        self.max_degree = int(edge_counts.max().item()) if len(keys) else 0
        self.source_ptr = torch.searchsorted(self.sources, torch.arange(atom_count + 1, device=self.device))

    def pair_range(self, atom_start, atom_stop):
        return self.source_ptr[[atom_start, atom_stop]].tolist()

    def coordinates(self, pair_ids):
        sources, targets = self.sources[pair_ids], self.targets[pair_ids]
        source_types, target_types = self.atom_types[sources], self.atom_types[targets]
        valid = self.type_masks[source_types, :, None] & self.type_masks[target_types, None, :]
        rows = self.offsets_gpu[sources, None] + self.type_ranks[source_types]
        columns = self.offsets_gpu[targets, None] + self.type_ranks[target_types]
        return rows[:, :, None].expand_as(valid), columns[:, None, :].expand_as(valid), valid

    def raw(self, local, pair_ids, phases):
        sources, targets = self.sources[pair_ids], self.targets[pair_ids]
        output = local.expand_nodes(sources).to(phases.dtype)
        output *= (sources == targets)[:, None, None]
        starts = self.edge_ptr[pair_ids]
        counts = self.edge_ptr[pair_ids + 1] - starts
        for rank in range(self.max_degree):
            selected = torch.nonzero(counts > rank, as_tuple=True)[0]
            edges = self.edge_order[starts[selected] + rank]
            hopping = local.expand_edges(edges)
            output[selected] += hopping * phases[edges, None, None]
        return output

    def batches(self, local, phases, start=0, stop=None):
        if stop is None:
            stop = len(self.sources)
        chunk_size = pair_chunk_size(self.full_norb, phases.element_size())
        for begin in range(start, stop, chunk_size):
            pair_ids = torch.arange(begin, min(begin + chunk_size, stop), device=self.device)
            values = self.raw(local, pair_ids, phases)
            reverse = self.raw(local, self.reverse[pair_ids], phases)
            values += reverse.mH
            rows, columns, valid = self.coordinates(pair_ids)
            yield rows, columns, valid, values

    def profiles(self, local_blocks, kpoints, device_window, lead_windows):
        size = int(device_window[1] - device_window[0])
        edge = torch.arange(1, size + 1, device=self.device)
        reverse_edge = edge.clone()
        extents = {tab: edge.new_ones(()) for tab in ("lead_L", "lead_R")}
        for kpoint in kpoints:
            phases = local_blocks[0].phases(kpoint)
            for local in local_blocks:
                for rows, columns, valid, values in self.batches(local, phases):
                    nonzero = valid & (values != 0)
                    device_rows = (rows >= device_window[0]) & (rows < device_window[1])
                    inside = nonzero & device_rows & (columns >= device_window[0]) & (columns < device_window[1])
                    row_indices = rows[inside] - device_window[0]
                    column_indices = columns[inside] - device_window[0]
                    edge.scatter_reduce_(0, row_indices, column_indices + 1, reduce="amax")
                    reverse_edge.scatter_reduce_(0, size - 1 - row_indices, size - column_indices, reduce="amax")
                    for tab, window in lead_windows.items():
                        selected = nonzero & device_rows & (columns >= window[0]) & (columns < window[1])
                        depths = rows - device_window[0] + 1 if tab == "lead_L" else device_window[1] - rows
                        maximum = torch.where(selected, depths, 0).amax()
                        extents[tab] = torch.maximum(extents[tab], maximum)
        return edge.cpu().numpy(), reverse_edge.cpu().numpy(), {
            tab: int(depth.item()) for tab, depth in extents.items()
        }


class CudaAssemblyPlan:
    """Stream pair batches into one BTD block row without element-sized plans."""

    def __init__(self, topology, device_window, blocks, lead_windows):
        self.topology = topology
        self.device_window = tuple(device_window)
        self.blocks = list(blocks)
        self.bounds = np.cumsum([device_window[0]] + list(blocks))
        self.lead_windows = lead_windows
        windows = list(zip(self.bounds[:-1], self.bounds[1:])) + list(lead_windows.values())
        atom_ranges = [
            (np.searchsorted(topology.offsets, start, side="right") - 1,
             np.searchsorted(topology.offsets, stop, side="left"))
            for start, stop in windows
        ]
        boundaries = torch.as_tensor(atom_ranges, device=topology.device)
        self.ranges = topology.source_ptr[boundaries].cpu().tolist()

    @staticmethod
    def put(output, rows, columns, valid, values, row_window, column_window):
        selected = (
            valid & (rows >= row_window[0]) & (rows < row_window[1])
            & (columns >= column_window[0]) & (columns < column_window[1])
        )
        output[rows[selected] - row_window[0], columns[selected] - column_window[0]] = values[selected]

    def write(self, cache, prefix, local, phases, k_index, factor, orthogonal, write_complex):
        kwargs = dict(dtype=phases.dtype, device=self.topology.device)
        workspace = torch.empty((3, max(self.blocks), max(self.blocks)), **kwargs)
        contacts, principal = {}, {}
        for tab, window in self.lead_windows.items():
            boundary = 0 if tab == "lead_L" else len(self.blocks) - 1
            width = window[1] - window[0]
            contacts[tab] = torch.zeros((self.blocks[boundary], width), **kwargs)
            principal[tab] = torch.zeros((width, width), **kwargs)
            if orthogonal:
                principal[tab].diagonal().fill_(1)
        violations = torch.zeros(2, dtype=torch.bool, device=self.topology.device)
        for block_index, size in enumerate(self.blocks):
            row_window = self.bounds[block_index:block_index + 2]
            neighbours = range(max(0, block_index - 1), min(len(self.blocks), block_index + 2))
            outputs = {
                column_block: workspace[slot, :size, :self.blocks[column_block]].zero_()
                for slot, column_block in enumerate(neighbours)
            }
            if orthogonal:
                outputs[block_index].diagonal().fill_(1)
            else:
                start, stop = self.ranges[block_index]
                for rows, columns, valid, values in self.topology.batches(local, phases, start, stop):
                    row_valid = valid & (rows >= row_window[0]) & (rows < row_window[1])
                    nonzero = row_valid & (values != 0)
                    inside = (columns >= self.device_window[0]) & (columns < self.device_window[1])
                    outside = (
                        (columns < self.bounds[max(0, block_index - 1)])
                        | (columns >= self.bounds[min(len(self.blocks), block_index + 2)])
                    )
                    violations[0] |= (nonzero & inside & outside).any()
                    for column_block, output in outputs.items():
                        self.put(output, rows, columns, row_valid, values, row_window,
                                 self.bounds[column_block:column_block + 2])
                    for tab, window in self.lead_windows.items():
                        boundary = 0 if tab == "lead_L" else len(self.blocks) - 1
                        if block_index == boundary:
                            self.put(contacts[tab], rows, columns, row_valid, values, row_window, window)
                        else:
                            violations[1] |= (
                                nonzero & (columns >= window[0]) & (columns < window[1])
                            ).any()
            for column_block, output in outputs.items():
                suffix = "d" if column_block == block_index else ("u" if column_block > block_index else "l")
                name = prefix + suffix
                write_complex(cache[name], f"{name}_k{k_index}_b{min(block_index, column_block)}",
                              output.mul_(factor))
        invalid_blocks, invalid_contacts = violations.tolist()
        if invalid_blocks:
            raise ValueError("Non-neighbouring block contribution would be lost.")
        if invalid_contacts:
            raise ValueError("Device-lead coupling extends beyond its boundary block.")
        if not orthogonal:
            for lead_index, (tab, window) in enumerate(self.lead_windows.items()):
                start, stop = self.ranges[len(self.blocks) + lead_index]
                for rows, columns, valid, values in self.topology.batches(local, phases, start, stop):
                    self.put(principal[tab], rows, columns, valid, values, window, window)
        return contacts, principal
