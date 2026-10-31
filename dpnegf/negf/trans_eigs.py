"""Contact-factor transmission spectra and spectral-normalized LCAO channels.

All matrices use the Green *coefficient* convention, including nonorthogonal
bases. No overlap is inserted into ``b_out.mH @ G_out_in @ b_in``.
Channel indices are zero based; spectra contain the explicit thin-SVD values.
"""
from dataclasses import dataclass
import math
import warnings

import torch


def _index_tensor(value, device, name):
    indices = torch.as_tensor(value, device=device)
    if indices.numel() and (indices.is_floating_point() or indices.is_complex()
                            or indices.dtype == torch.bool):
        raise ValueError(f"{name} must contain integer indices")
    return indices.to(torch.long)


def factor_broadening(gamma, *, psd_atol=0., psd_rtol=None,
                      rank_atol=0., rank_rtol=None):
    """Return a thin PSD factor and diagnostics, without hiding negative Gamma.

    Absolute thresholds have the same energy unit as Gamma. Defaults scale
    with dtype roundoff and contact dimension; PSD and rank tolerances are
    independent. Truncation of small Gamma eigenvalues is not a bound on the
    transmission error (in particular at narrow resonances).
    """
    if gamma.ndim != 2 or gamma.shape[0] != gamma.shape[1]:
        raise ValueError("Contact broadening must be a square 2-D tensor")
    if not torch.isfinite(gamma).all():
        raise ValueError("Non-finite contact broadening")
    eps = torch.finfo(gamma.real.dtype).eps
    if psd_rtol is None:
        psd_rtol = 100 * max(1, len(gamma)) * eps
    if rank_rtol is None:
        rank_rtol = max(1, len(gamma)) * eps
    if (not all(math.isfinite(x) for x in (psd_atol, psd_rtol, rank_atol, rank_rtol))
            or min(psd_atol, psd_rtol, rank_atol, rank_rtol) < 0):
        raise ValueError("Broadening tolerances must be finite and nonnegative")
    scale = torch.linalg.matrix_norm(gamma)
    hermitian_error = torch.linalg.matrix_norm(gamma - gamma.mH)
    if hermitian_error > psd_atol + psd_rtol * scale:
        raise ValueError("Contact broadening is significantly non-Hermitian")
    values, vectors = torch.linalg.eigh((gamma + gamma.mH) / 2)
    spectral_scale = values.abs().max() if values.numel() else scale
    negative_tol = torch.maximum(values.new_tensor(psd_atol), psd_rtol * spectral_scale)
    if (values < -negative_tol).any():
        raise ValueError("Contact broadening is not positive semidefinite")
    maximum = values.clamp_min(0).max() if values.numel() else scale
    threshold = torch.maximum(values.new_tensor(rank_atol), rank_rtol * maximum)
    keep = values > threshold
    discarded = values[(values > 0) & ~keep]
    factor = vectors[:, keep] * values[keep].sqrt()
    return factor, {
        "hermitian_residual": hermitian_error,
        "minimum_eigenvalue": values.min() if values.numel() else scale,
        "negative_tolerance": negative_tol, "rank_threshold": threshold,
        "discarded_positive_count": discarded.numel(),
        "discarded_positive_sum": discarded.sum(),
        "discarded_positive_max": discarded.max() if discarded.numel() else values.new_zeros(()),
        "factor_residual": torch.linalg.matrix_norm(gamma - factor @ factor.mH),
    }


@dataclass
class TransportContext:
    """One energy/k/spin block in the existing RGF boundary-contact layout.

    Optional contact indices are local to the first/last RGF block and follow
    the rows of the supplied self-energy. Cross-block contacts are rejected.
    ``orbital_map[i]`` is the original device orbital index of block-order row i.
    ``gr_left`` may reuse the left-connected Green blocks from full RGF.
    ``g_trans`` is the full first-to-last block G_LR (rows in the first
    block, columns in the last), reusable directly for right incidence.
    """
    energy: object
    hd: list
    sd: list
    hu: list
    hl: list
    su: list
    sl: list
    left_se: torch.Tensor
    right_se: torch.Tensor
    eta_device: float = 0.
    E_ref: float = 0.
    kpoint: object = None
    spin: object = None
    eta_lead: object = None
    left_indices: object = None
    right_indices: object = None
    orbital_map: object = None
    gr_left: object = None
    g_trans: object = None

    def __post_init__(self):
        if not self.hd or len(self.sd) != len(self.hd):
            raise ValueError("Invalid diagonal block lists")
        if any(len(x) != len(self.hd) - 1 for x in (self.hu, self.hl, self.su, self.sl)):
            raise ValueError("Invalid off-diagonal block lists")
        if not self.hd[0].is_complex():
            raise ValueError("Transport blocks must use a complex dtype, as in the RGF solver")
        reference = self.hd[0]
        sizes = []
        for h, s in zip(self.hd, self.sd):
            if h.ndim != 2 or h.shape[0] == 0 or h.shape[0] != h.shape[1] or s.shape != h.shape:
                raise ValueError("Diagonal H/S blocks must be matching nonempty square matrices")
            sizes.append(h.shape[0])
        for i in range(len(sizes) - 1):
            shape = (sizes[i], sizes[i + 1])
            if (self.hu[i].shape != shape or self.su[i].shape != shape
                    or self.hl[i].shape != shape[::-1] or self.sl[i].shape != shape[::-1]):
                raise ValueError("Off-diagonal H/S blocks have incompatible shapes")
        matrices = [x for parts in (self.hd, self.sd, self.hu, self.hl, self.su, self.sl)
                    for x in parts] + [self.left_se, self.right_se]
        if self.gr_left is not None:
            if len(self.gr_left) != len(sizes) or any(
                    g.shape != (n, n) for g, n in zip(self.gr_left, sizes)):
                raise ValueError("Cached gr_left blocks have incompatible shapes")
            matrices = matrices + list(self.gr_left)
        if self.g_trans is not None:
            if self.g_trans.shape != (sizes[0], sizes[-1]):
                raise ValueError("Cached g_trans must be the full boundary block G_LR")
            matrices.append(self.g_trans)
        if any(x.device != reference.device or x.dtype != reference.dtype for x in matrices):
            raise ValueError("Transport matrices and cached Green blocks must share dtype and device")
        self.z = torch.as_tensor(self.energy, device=self.hd[0].device,
                                 dtype=self.hd[0].dtype).reshape(()) + self.E_ref + 1j * self.eta_device
        if any(not torch.isfinite(x).all() for x in matrices + [self.z]):
            raise ValueError("Non-finite transport input")
        self.left_indices = self._indices(self.left_se, self.hd[0], self.left_indices, False)
        self.right_indices = self._indices(self.right_se, self.hd[-1], self.right_indices, True)
        n = sum(x.shape[0] for x in self.hd)
        if self.orbital_map is None:
            self.orbital_map = torch.arange(n, device=self.hd[0].device)
        self.orbital_map = _index_tensor(self.orbital_map, self.hd[0].device, "orbital_map")
        if self.orbital_map.shape != (n,) or not torch.equal(
                self.orbital_map.sort().values, torch.arange(n, device=self.hd[0].device)):
            raise ValueError("orbital_map must permute all original device orbitals")

    @staticmethod
    def _indices(se, block, indices, right):
        if se.ndim != 2 or se.shape[0] != se.shape[1] or se.shape[0] > block.shape[0]:
            raise ValueError("Self-energy must fit in its boundary block; cross-block contacts are unsupported")
        if indices is None:
            start = block.shape[0] - se.shape[0] if right else 0
            indices = torch.arange(start, start + se.shape[0], device=block.device)
        indices = _index_tensor(indices, block.device, "Boundary contact indices")
        if (indices.shape != (se.shape[0],) or indices.unique().numel() != indices.numel()
                or (indices < 0).any() or (indices >= block.shape[0]).any()):
            raise ValueError("Invalid boundary contact indices")
        return indices

    def diagonal(self, i):
        d = self.z * self.sd[i] - self.hd[i]
        if i == 0:
            idx = self.left_indices
            d[idx[:, None], idx[None, :]] -= self.left_se
        if i == len(self.hd) - 1:
            idx = self.right_indices
            d[idx[:, None], idx[None, :]] -= self.right_se
        return d


class _ContactSolver:
    """Block elimination, keeping factors only when channels are requested."""
    def __init__(self, context, incoming, retain):
        self.c = context
        self.reverse = incoming == "right"
        self.order = list(range(len(context.hd)))
        if self.reverse:
            self.order.reverse()
        self.in_idx = context.right_indices if self.reverse else context.left_indices
        self.out_idx = context.left_indices if self.reverse else context.right_indices
        self.retain = retain
        self.factors = []
        self.transfers = []

    def upper(self, i):
        c = self.c
        q = self.order[i + 1] if self.reverse else self.order[i]
        return c.z * (c.sl[q] if self.reverse else c.su[q]) - (c.hl[q] if self.reverse else c.hu[q])

    def lower(self, i):
        c = self.c
        q = self.order[i] if self.reverse else self.order[i - 1]
        return c.z * (c.su[q] if self.reverse else c.sl[q]) - (c.hu[q] if self.reverse else c.hl[q])

    def solve(self, contact_source, *, blocks=False):
        c = self.c
        previous = None
        transfer = None
        responses = []
        cached = bool(self.factors)
        for i, q in enumerate(self.order):
            rhs = c.hd[q].new_zeros((c.hd[q].shape[0], contact_source.shape[1]))
            if i == 0:
                rhs[self.in_idx] = contact_source
            else:
                lower = self.lower(i)
                rhs = rhs - lower @ previous
            if cached:
                factor, pivots = self.factors[i]
                response = torch.linalg.lu_solve(factor, pivots, rhs)
                transfer = self.transfers[i] if i < len(self.order) - 1 else None
            elif c.gr_left is not None and not self.reverse:
                response = c.gr_left[q] @ rhs
                transfer = c.gr_left[q] @ self.upper(i) if i < len(self.order) - 1 else None
            else:
                d = c.diagonal(q)
                if i:
                    d = d - lower @ transfer
                factor, pivots = torch.linalg.lu_factor(d)
                response = torch.linalg.lu_solve(factor, pivots, rhs)
                transfer = torch.linalg.lu_solve(factor, pivots, self.upper(i)) if i < len(self.order) - 1 else None
                if self.retain:
                    self.factors.append((factor, pivots))
                    if transfer is not None:
                        self.transfers.append(transfer)
            if blocks:
                responses.append(response)
            previous = response
        if not blocks:
            return previous[self.out_idx]
        for i in range(len(responses) - 2, -1, -1):
            if self.transfers:
                transfer = self.transfers[i]
            else:
                transfer = c.gr_left[self.order[i]] @ self.upper(i)
            responses[i] = responses[i] - transfer @ responses[i + 1]
        return list(reversed(responses)) if self.reverse else responses


def _compute(context, *, incoming="left", channels=False, channel_indices=None,
             n_channels=None, return_format="blocks", **factor_options):
    if incoming not in ("left", "right"):
        raise ValueError("incoming must be 'left' or 'right'")
    if channel_indices is not None and n_channels is not None:
        raise ValueError("channel_indices and n_channels are mutually exclusive")
    if return_format not in ("blocks", "original"):
        raise ValueError("return_format must be 'blocks' or 'original'")
    c = context
    se_in, se_out = (c.left_se, c.right_se) if incoming == "left" else (c.right_se, c.left_se)
    gamma_in, gamma_out = 1j * (se_in - se_in.mH), 1j * (se_out - se_out.mH)
    b_in, diag_in = factor_broadening(gamma_in, **factor_options)
    b_out, diag_out = factor_broadening(gamma_out, **factor_options)
    solver = _ContactSolver(c, incoming, retain=channels)
    # Right incidence uses G_LR already computed by full RGF. Select the actual
    # contact rows/columns, including reduced or non-contiguous contacts.
    if incoming == "right" and c.g_trans is not None:
        green = c.g_trans[c.left_indices[:, None], c.right_indices[None, :]]
    else:
        # Contact identity also retains discarded Gamma directions for raw Caroli.
        green = solver.solve(torch.eye(len(gamma_in), device=se_in.device, dtype=se_in.dtype))
    k = b_out.mH @ green @ b_in
    if not torch.isfinite(green).all() or not torch.isfinite(k).all():
        raise ValueError("Non-finite contact Green function or transmission matrix")
    if min(k.shape) == 0:
        singular = gamma_in.real.new_empty(0)
        vh = b_in.new_empty((0, b_in.shape[1]))
    elif channels:
        _, singular, vh = torch.linalg.svd(k, full_matrices=False)
    else:
        singular = torch.linalg.svdvals(k)
    tau = singular.square()
    raw = torch.trace(gamma_out @ green @ gamma_in @ green.mH).real
    if not torch.isfinite(tau).all() or not torch.isfinite(raw):
        raise ValueError("Non-finite transmission spectrum or Caroli value")
    check_tol = 100 * max(1, len(gamma_in), len(gamma_out)) * torch.finfo(tau.dtype).eps
    error = (tau.sum() - raw).abs()
    tolerance = check_tol * (1 + torch.maximum(tau.sum().abs(), raw.abs()))
    range_tol = 100 * max(len(c.hd), len(gamma_in), len(gamma_out), 1) * torch.finfo(tau.dtype).eps
    diagnostics = {"input_broadening": diag_in, "output_broadening": diag_out,
                   "transmission_absolute_error": error,
                   "transmission_tolerance": tolerance,
                   "transmission_check_passed": error <= tolerance,
                   "dtype": str(k.dtype),
                   "tau_range_tolerance": range_tol,
                   "tau_above_one": tau > 1 + range_tol,
                   "numerical_zero_singular_values": singular == 0}
    if diagnostics["tau_above_one"].any():
        warnings.warn("Transmission eigenvalues exceed one; inspect self-energies and eta convergence", RuntimeWarning)
    result = dict(energy=torch.as_tensor(c.energy).real, E_ref=c.E_ref,
                  kpoint=c.kpoint, spin=c.spin, eta_device=c.eta_device, eta_lead=c.eta_lead,
                  incoming=incoming, outgoing="right" if incoming == "left" else "left",
                  tau_all=tau, singular_values=singular, log_tau=2 * singular.log(),
                  T_spectral=tau.sum(), T_caroli_raw=raw,
                  input_rank=b_in.shape[1], output_rank=b_out.shape[1],
                  n_implicit_zero_inputs=max(b_in.shape[1] - b_out.shape[1], 0),
                  normalization="spectral", orbital_map=c.orbital_map,
                  basis_order="blocks", diagnostics=diagnostics)
    if not channels:
        return result
    if n_channels is not None:
        if isinstance(n_channels, bool) or not isinstance(n_channels, int) or n_channels < 0:
            raise ValueError("n_channels must be a nonnegative integer")
        indices = torch.arange(min(n_channels, len(tau)), device=tau.device)
    elif channel_indices is None:
        indices = torch.arange(len(tau), device=tau.device)
    else:
        indices = torch.as_tensor(channel_indices, device=tau.device)
        if indices.numel() == 0:
            indices = indices.to(torch.long)
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("channel_indices must be a 1-D integer sequence")
        if (indices < 0).any() or (indices >= len(tau)).any() or indices.unique().numel() != indices.numel():
            raise ValueError("Invalid or repeated channel index")
    source = b_in @ vh.mH[:, indices]
    coeff = solver.solve(source, blocks=True)
    if any(not torch.isfinite(x).all() for x in coeff):
        raise ValueError("Non-finite eigenchannel coefficients")
    out = coeff[-1][c.right_indices] if incoming == "left" else coeff[0][c.left_indices]
    target = torch.diag(tau[indices]).to(out.dtype)
    diagnostics["channel_raw_residual"] = torch.linalg.matrix_norm(out.mH @ gamma_out @ out - target)
    diagnostics["channel_kept_residual"] = torch.linalg.matrix_norm((b_out.mH @ out).mH @ (b_out.mH @ out) - target)
    residual_sq = tau.new_zeros(())
    matrix_sq = tau.new_zeros(())
    for i, ci in enumerate(coeff):
        d = c.diagonal(i)
        residual = d @ ci
        matrix_sq += d.abs().square().sum()
        if i:
            lower = c.z * c.sl[i - 1] - c.hl[i - 1]
            residual += lower @ coeff[i - 1]
            matrix_sq += lower.abs().square().sum()
        if i + 1 < len(coeff):
            upper = c.z * c.su[i] - c.hu[i]
            residual += upper @ coeff[i + 1]
            matrix_sq += upper.abs().square().sum()
        if i == (0 if incoming == "left" else len(coeff) - 1):
            residual[solver.in_idx] -= source
        residual_sq += residual.abs().square().sum()
    denominator = matrix_sq.sqrt() * sum(x.abs().square().sum() for x in coeff).sqrt() + torch.linalg.matrix_norm(source)
    diagnostics["solve_relative_residual"] = residual_sq.sqrt() / denominator.clamp_min(torch.finfo(tau.dtype).tiny)
    if return_format == "original":
        coeff = torch.cat(coeff)[torch.argsort(c.orbital_map)]
        result["basis_order"] = "original"
    result.update(coefficients=coeff, selected_indices=indices,
                  tau_selected=tau[indices], block_sizes=[len(x) for x in c.hd])
    return result


def transmission_eigenvalues(context, *, incoming="left", **factor_options):
    """Compute a descending thin transmission spectrum, without channel SVD vectors."""
    return _compute(context, incoming=incoming, **factor_options)


def transmission_eigenchannels(context, *, incoming="left", channel_indices=None,
                              n_channels=None, return_format="blocks", **factor_options):
    """Reconstruct selected spectral-normalized complex LCAO coefficient columns."""
    return _compute(context, incoming=incoming, channels=True,
                    channel_indices=channel_indices, n_channels=n_channels,
                    return_format=return_format, **factor_options)
