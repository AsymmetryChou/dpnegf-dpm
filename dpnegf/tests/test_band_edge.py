import copy
import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from dargs.dargs import Argument, ArgumentTypeError
from dptb.nn.build import build_model

from dpnegf.runner.NEGF import NEGF
from dpnegf.utils.argcheck import normalize_run
from dpnegf.utils.band_edge import calculate_band_edges, validate_fermi_in_band_gap
from dpnegf.utils.elec_struc_cal import ElecStruCal


ROOT = Path(__file__).resolve().parents[2]
HBN_ROOT = ROOT / "examples" / "hBN"
HBN_CHECKPOINT = HBN_ROOT / "train" / "train_out" / "checkpoint" / "nnsk.ep3000.pth"
HBN_PRIMITIVE = HBN_ROOT / "train" / "data" / "POSCAR"
HBN_NEGF_STRUCTURE = HBN_ROOT / "stru_negf.xyz"
HBN_EIGENVALUES = HBN_ROOT / "train" / "data" / "kpath.0" /"eigenvalues.npy"


def test_hbn_saved_eigenvalues_band_edges():
    eigenvalues = np.load(HBN_EIGENVALUES)[0]

    e_c, e_v = calculate_band_edges(
        eigenvalues=eigenvalues,
        neutral_electrons=8,
        spin_degeneracy=2,
    )

    assert e_v == pytest.approx(-6.0866060, abs=1e-5)
    assert e_c == pytest.approx(-1.4251115, abs=1e-5)
    assert e_c - e_v == pytest.approx(4.6614945, abs=1e-5)


def test_band_edges_use_global_kpoint_extrema():
    eigenvalues = np.array(
        [
            [-5.0, -2.0, 2.0, 4.0],
            [-4.0, -1.0, 3.0, 5.0],
            [-6.0, -3.0, 1.0, 6.0],
        ]
    )

    e_c, e_v = calculate_band_edges(
        eigenvalues,
        neutral_electrons=4,
        spin_degeneracy=2,
    )

    assert e_v == -1.0
    assert e_c == 1.0


@pytest.mark.parametrize(
    "neutral_electrons, spin_degeneracy, match",
    [
        (3, 2, "integer number of occupied bands"),
        (0, 2, "outside the available spectrum"),
        (8, 2, "outside the available spectrum"),
    ],
)
def test_invalid_occupied_band_count_raises(
    neutral_electrons,
    spin_degeneracy,
    match,
):
    eigenvalues = np.array([[-2.0, -1.0, 1.0, 2.0]])

    with pytest.raises(ValueError, match=match):
        calculate_band_edges(
            eigenvalues,
            neutral_electrons,
            spin_degeneracy,
        )


def test_overlapping_bands_raise():
    eigenvalues = np.array(
        [
            [-2.0, 0.5, 1.0],
            [-1.0, 0.0, 0.25],
        ]
    )

    with pytest.raises(ValueError, match="not semiconducting"):
        calculate_band_edges(
            eigenvalues,
            neutral_electrons=4,
            spin_degeneracy=2,
        )


@pytest.mark.parametrize("e_fermi", [-1.0, 2.0, -2.0, 3.0])
def test_fermi_level_must_be_strictly_inside_gap(e_fermi):
    with pytest.raises(
        ValueError,
        match=r"lead_L.*E_v=.*E_f=.*E_c=",
    ):
        validate_fermi_in_band_gap(
            e_fermi=e_fermi,
            e_c=2.0,
            e_v=-1.0,
            lead_name="lead_L",
        )


def test_compute_band_edges_schema_default_and_override():
    with open(HBN_ROOT / "negf.json") as fp:
        config = json.load(fp)

    config["structure"] = str(HBN_NEGF_STRUCTURE)

    normalized = normalize_run(copy.deepcopy(config))
    assert (
        normalized["task_options"]["stru_options"]["compute_band_edges"]
        is False
    )

    config["task_options"]["stru_options"]["compute_band_edges"] = True
    normalized = normalize_run(config)
    assert (
        normalized["task_options"]["stru_options"]["compute_band_edges"]
        is True
    )

    config["task_options"]["stru_options"]["compute_band_edges"] = "yes"
    with pytest.raises(ArgumentTypeError):
        normalize_run(config)


def test_hbn_electronic_structure_returns_fermi_and_band_edges():
    model = build_model(checkpoint=str(HBN_CHECKPOINT))
    calculator = ElecStruCal(model=model, device="cpu")

    _, e_fermi, e_c, e_v = calculator.get_fermi_level(
        data=str(HBN_PRIMITIVE),
        nel_atom={"B": 3, "N": 5},
        meshgrid=[1, 10, 1],
        eig_solver="torch",
        compute_band_edges=True,
    )

    assert e_v == pytest.approx(-6.5426044, abs=1e-5)
    assert e_fermi == pytest.approx(-2.8933584, abs=1e-5)
    assert e_c == pytest.approx(-0.8259304, abs=1e-5)

    validate_fermi_in_band_gap(e_fermi, e_c, e_v)

    default_result = calculator.get_fermi_level(
        data=str(HBN_PRIMITIVE),
        nel_atom={"B": 3, "N": 5},
        klist=np.array([[0.0, 0.0, 0.0]]),
        eig_solver="torch",
    )

    assert len(default_result) == 2


def test_hbn_negf_initialization_reports_both_lead_band_edges(
    tmp_path,
    caplog,
):
    with open(HBN_ROOT / "negf.json") as fp:
        config = json.load(fp)

    config["structure"] = str(HBN_NEGF_STRUCTURE)

    stru_options = config["task_options"]["stru_options"]
    stru_options["compute_band_edges"] = True

    # Preserve the hBN example and its real model while keeping this
    # integration test focused on feature wiring rather than dense
    # k-mesh convergence.
    stru_options["lead_L"]["kmesh_lead_Ef"] = [1, 2, 1]
    stru_options["lead_R"]["kmesh_lead_Ef"] = [1, 2, 1]

    config["task_options"]["emin"] = -1.0
    config["task_options"]["emax"] = 1.0
    config["task_options"]["espacing"] = 1.0
    config["task_options"]["output_options"] = {"tc": True}
    config["task_options"]["self_energy_options"] = {
        "solver": "Sancho-Rubio",
        "numba_jit": False,
        "parallel": {
            "n_workers": 1,
            "blas_threads": 1,
        },
    }

    task_options = normalize_run(config)["task_options"]
    task_options.pop("task")

    results_path = tmp_path / "results"
    results_path.mkdir()

    model = build_model(checkpoint=str(HBN_CHECKPOINT))

    with caplog.at_level(logging.INFO, logger="dpnegf.runner.NEGF"):
        negf = NEGF(
            model=model,
            structure=str(HBN_NEGF_STRUCTURE),
            results_path=str(results_path),
            AtomicData_options=normalize_run(config)["AtomicData_options"],
            **task_options,
        )

    for lead in ("lead_L", "lead_R"):
        assert negf.E_v[lead] < negf.e_fermi[lead] < negf.E_c[lead]
        assert negf.out["E_v"][lead] == negf.E_v[lead]
        assert negf.out["E_c"][lead] == negf.E_c[lead]
        assert f"Band edges for {lead}:" in caplog.text

    assert negf.E_v["lead_L"] == pytest.approx(
        negf.E_v["lead_R"],
        abs=1e-5,
    )
    assert negf.E_c["lead_L"] == pytest.approx(
        negf.E_c["lead_R"],
        abs=1e-5,
    )

    assert caplog.text.index(
        "Fermi level for lead_R:"
    ) < caplog.text.index(
        "Band edges for lead_L:"
    )
    assert caplog.text.index(
        "Band edges for lead_L:"
    ) < caplog.text.index(
        "Band edges for lead_R:"
    )

    negf.compute()

    output = torch.load(
        results_path / "negf.out.pth",
        weights_only=False,
    )

    assert output["E_v"] == pytest.approx(negf.E_v)
    assert output["E_c"] == pytest.approx(negf.E_c)


def test_hbn_negf_rejects_explicit_fermi_outside_gap(
    tmp_path,
    monkeypatch,
):
    with open(HBN_ROOT / "negf.json") as fp:
        config = json.load(fp)

    config["structure"] = str(HBN_NEGF_STRUCTURE)
    config["task_options"]["stru_options"]["compute_band_edges"] = True
    config["task_options"]["e_fermi"] = 0.0

    task_options = normalize_run(config)["task_options"]
    task_options.pop("task")

    monkeypatch.setattr(
        NEGF,
        "_calculate_lead_electronic_structure",
        lambda self, **kwargs: (-2.9, -0.8, -6.5),
    )

    model = build_model(checkpoint=str(HBN_CHECKPOINT))

    with pytest.raises(
        ValueError,
        match=r"lead_L.*E_v=-6.5.*E_f=0.0.*E_c=-0.8",
    ):
        NEGF(
            model=model,
            structure=str(HBN_NEGF_STRUCTURE),
            results_path=str(tmp_path),
            AtomicData_options=normalize_run(config)["AtomicData_options"],
            **task_options,
        )