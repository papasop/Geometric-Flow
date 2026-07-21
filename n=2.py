#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pulser_n2_path_order_minimal_clock4.py

N=2 Pulser/Qutip minimal path-ordering memory diagnostic, with explicit
4 ns clock quantization.

Question tested
---------------
Do two two-segment detuning schedules with the same total duration,
same Rabi pulse-area proxy, and same weighted average detuning produce
different final states solely because the segment order is reversed?

    forward:  Delta_1 -> Delta_2
    reverse:  Delta_2 -> Delta_1

This is the minimal two-qubit adaptation for Hamiltonian-learning discussion:
a time-independent learned generator U=exp(-i H tau), or an average-Hamiltonian
model, can miss the BCH commutator contribution

    [H(Delta_1), H(Delta_2)] = (Delta_1 - Delta_2) [H_X, N].

What changed in this clock4 version
-----------------------------------
Pulser DigitalAnalogDevice uses a 4 ns channel clock. This version quantizes
all durations before building the sequence, so printed durations, average
detuning, pulse-area proxy, and the actually executed Pulser sequence agree.
No Pulser duration-rounding warnings should appear.

Scope
-----
- Local Pulser/Qutip exact-state simulation only.
- Not PASQAL QPU data.
- Not tomography.
- Not a direct K=1 / detG signature-switching test.
- Hardware can directly test output probabilities/counts; full statevector
  fidelity requires simulation or tomography.

Colab install
-------------
!pip install -q -U pulser==1.8.0 pulser-simulation==1.8.0 qutip pandas numpy matplotlib

Run
---
%run pulser_n2_path_order_minimal_clock4.py

Outputs
-------
pulser_n2_path_order_minimal_clock4/
  n2_pair_metrics_clock4.csv
  n2_schedule_metrics_clock4.csv
  n2_certificate_clock4.json
  n2_distribution_plot_clock4.png
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pulser import Pulse, Sequence, Register
from pulser.devices import DigitalAnalogDevice
from pulser.waveforms import ConstantWaveform
from pulser_simulation import QutipEmulator


# =============================================================================
# CONFIG
# =============================================================================

OUTDIR = Path("pulser_n2_path_order_minimal_clock4")
OUTDIR.mkdir(exist_ok=True)

N = 2
SPACING_UM = 8.0

OMEGA = 1.22
TOTAL_LOOP = 2.22

BASE_DETUNING = -0.31

DELTA_1 = -0.38
DELTA_2 = -0.25
FRAC_1 = 0.35

CLOCK_NS = 4
MIN_DURATION_NS = 16
MAX_DURATION_NS = 10000

PAIR_CSV = OUTDIR / "n2_pair_metrics_clock4.csv"
SCHEDULE_CSV = OUTDIR / "n2_schedule_metrics_clock4.csv"
CERT_JSON = OUTDIR / "n2_certificate_clock4.json"
PLOT_PATH = OUTDIR / "n2_distribution_plot_clock4.png"


# =============================================================================
# CLOCK / DURATION HELPERS
# =============================================================================

def ceil_to_clock_ns(duration_ns: int, clock_ns: int = CLOCK_NS) -> int:
    """Round up to the device clock period."""
    return int(math.ceil(int(duration_ns) / clock_ns) * clock_ns)


def clamp_to_clock_duration(duration_ns: int) -> int:
    """Clamp duration and round up to clock. Assumes min/max are clock-compatible."""
    duration_ns = max(MIN_DURATION_NS, min(MAX_DURATION_NS, int(duration_ns)))
    return ceil_to_clock_ns(duration_ns, CLOCK_NS)


def raw_duration_from_loop_ns(n: int = N, omega: float = OMEGA, loop: float = TOTAL_LOOP) -> int:
    # Same convention as the larger scan:
    # Phi_model = 0.5 * sqrt(N) * Omega * T_us
    t_us = 2 * math.pi * loop / (math.sqrt(n) * omega)
    return int(round(1000 * t_us))


def duration_from_loop_ns(n: int = N, omega: float = OMEGA, loop: float = TOTAL_LOOP) -> int:
    return clamp_to_clock_duration(raw_duration_from_loop_ns(n, omega, loop))


def forward_segment_durations(frac1: float = FRAC_1) -> tuple[int, int, int]:
    """
    Return clock-aligned forward durations d1, d2, total.

    Important: total is clock-aligned first; d1 is then rounded up to clock;
    d2 = total - d1 is automatically clock-aligned because total and d1 are.
    """
    total = duration_from_loop_ns()
    raw_d1 = int(round(total * frac1))
    d1 = ceil_to_clock_ns(raw_d1, CLOCK_NS)
    d1 = max(MIN_DURATION_NS, min(total - MIN_DURATION_NS, d1))
    # Keep d1 a clock multiple after clamping.
    d1 = ceil_to_clock_ns(d1, CLOCK_NS)
    if d1 >= total:
        d1 = total - MIN_DURATION_NS
    d2 = total - d1
    if d2 < MIN_DURATION_NS:
        d2 = MIN_DURATION_NS
        d1 = total - d2
    assert d1 % CLOCK_NS == 0
    assert d2 % CLOCK_NS == 0
    assert total % CLOCK_NS == 0
    assert d1 + d2 == total
    return int(d1), int(d2), int(total)


def weighted_avg_detuning(delta1: float, delta2: float, duration1_ns: int, duration2_ns: int) -> float:
    total = duration1_ns + duration2_ns
    return float((delta1 * duration1_ns + delta2 * duration2_ns) / total)


# =============================================================================
# BASIC HELPERS
# =============================================================================

def header(text: str) -> None:
    print()
    print("=" * 100)
    print(text)
    print("=" * 100)


def state_labels(n: int) -> list[str]:
    return [format(i, f"0{n}b") for i in range(2**n)]


def normalize_state(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=np.complex128).ravel()
    nrm = np.linalg.norm(psi)
    if nrm <= 0:
        raise ValueError("Zero statevector norm")
    return psi / nrm


def normalize_prob(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float).ravel()
    s = float(np.sum(p))
    if s <= 0:
        raise ValueError("Non-positive probability sum")
    return p / s


def corrected_statevector(psi: np.ndarray, n: int) -> np.ndarray:
    """
    Same basis-label correction used in the larger V2.2 script.
    It applies a consistent global bit flip to every state.
    Fidelity is invariant under the same permutation, but probability labels
    become aligned with the sampled final-state convention used previously.
    """
    psi = normalize_state(psi)
    dim = 2**n
    if len(psi) != dim:
        raise ValueError(f"len(psi)={len(psi)} != 2^N={dim}")

    out = np.zeros_like(psi)
    for i in range(dim):
        b = format(i, f"0{n}b")
        flipped = "".join("1" if c == "0" else "0" for c in b)
        j = int(flipped, 2)
        out[j] = psi[i]
    return normalize_state(out)


def probs_from_state(psi: np.ndarray) -> np.ndarray:
    return normalize_prob(np.abs(normalize_state(psi)) ** 2)


def nominal_coords(n: int = N, spacing: float = SPACING_UM) -> np.ndarray:
    return np.array(
        [[(i - (n - 1) / 2) * spacing, 0.0] for i in range(n)],
        dtype=float,
    )


def make_register(coords: np.ndarray) -> Register:
    return Register({f"q{i}": np.array(coords[i], dtype=float) for i in range(len(coords))})


def add_constant_pulse(seq: Sequence, omega: float, detuning: float, duration_ns: int, phase: float = 0.0) -> None:
    if duration_ns % CLOCK_NS != 0:
        raise ValueError(f"duration_ns={duration_ns} is not a multiple of CLOCK_NS={CLOCK_NS}")
    omega_wf = ConstantWaveform(duration_ns, omega)
    det_wf = ConstantWaveform(duration_ns, detuning)
    seq.add(Pulse(omega_wf, det_wf, phase), "rydberg_global")


# =============================================================================
# SEQUENCE BUILDERS
# =============================================================================

def build_constant_sequence(detuning: float, duration_total_ns: int, coords: np.ndarray | None = None):
    if coords is None:
        coords = nominal_coords()

    if duration_total_ns % CLOCK_NS != 0:
        raise ValueError("duration_total_ns must be clock-aligned")

    reg = make_register(coords)
    seq = Sequence(reg, DigitalAnalogDevice)
    seq.declare_channel("rydberg_global", "rydberg_global")
    add_constant_pulse(seq, OMEGA, detuning, duration_total_ns)

    meta = {
        "family": "constant",
        "detuning_1": float(detuning),
        "detuning_2": np.nan,
        "duration_1_ns": int(duration_total_ns),
        "duration_2_ns": 0,
        "duration_total_ns": int(duration_total_ns),
        "frac_1": 1.0,
        "avg_detuning": float(detuning),
        "pulse_area_proxy": float(OMEGA * duration_total_ns / 1000.0),
        "clock_ns": CLOCK_NS,
    }
    return seq, meta


def build_two_segment_sequence(
    delta1: float,
    delta2: float,
    duration1_ns: int,
    duration2_ns: int,
    coords: np.ndarray | None = None,
):
    if coords is None:
        coords = nominal_coords()

    if duration1_ns % CLOCK_NS != 0 or duration2_ns % CLOCK_NS != 0:
        raise ValueError("segment durations must be clock-aligned")

    total = duration1_ns + duration2_ns
    avg = weighted_avg_detuning(delta1, delta2, duration1_ns, duration2_ns)

    reg = make_register(coords)
    seq = Sequence(reg, DigitalAnalogDevice)
    seq.declare_channel("rydberg_global", "rydberg_global")
    add_constant_pulse(seq, OMEGA, delta1, duration1_ns)
    add_constant_pulse(seq, OMEGA, delta2, duration2_ns)

    meta = {
        "family": "two_segment",
        "detuning_1": float(delta1),
        "detuning_2": float(delta2),
        "duration_1_ns": int(duration1_ns),
        "duration_2_ns": int(duration2_ns),
        "duration_total_ns": int(total),
        "frac_1": float(duration1_ns / total),
        "avg_detuning": float(avg),
        "pulse_area_proxy": float(OMEGA * total / 1000.0),
        "clock_ns": CLOCK_NS,
    }
    return seq, meta


def get_final_state_array(result) -> np.ndarray:
    attempts = []

    if hasattr(result, "get_final_state"):
        attempts.append(lambda: result.get_final_state())
        attempts.append(lambda: result.get_final_state(reduce_to_basis="ground-rydberg"))
        attempts.append(lambda: result.get_final_state(ignore_global_phase=True))

    if hasattr(result, "states") and len(result.states) > 0:
        attempts.append(lambda: result.states[-1])

    if hasattr(result, "_states") and len(result._states) > 0:
        attempts.append(lambda: result._states[-1])

    last_err = None
    for call in attempts:
        try:
            state = call()
            if hasattr(state, "full"):
                return np.asarray(state.full()).ravel()
            return np.asarray(state).ravel()
        except Exception as exc:
            last_err = exc

    raise RuntimeError(f"Could not access final state. Last error: {repr(last_err)}")


def final_statevector_from_sequence(seq: Sequence, n: int = N) -> np.ndarray:
    sim = QutipEmulator.from_sequence(seq)
    result = sim.run()
    arr = get_final_state_array(result)
    arr = normalize_state(arr)

    if len(arr) != 2**n:
        raise RuntimeError(f"Final state dimension {len(arr)} != 2^N={2**n}")

    return corrected_statevector(arr, n)


def build_all_schedules():
    coords = nominal_coords()
    d1, d2, total = forward_segment_durations()

    avg_det = weighted_avg_detuning(DELTA_1, DELTA_2, d1, d2)

    builders = {
        "base": lambda: build_constant_sequence(BASE_DETUNING, total, coords),
        "avg": lambda: build_constant_sequence(avg_det, total, coords),
        # forward uses d1, d2
        "forward": lambda: build_two_segment_sequence(DELTA_1, DELTA_2, d1, d2, coords),
        # reverse uses exactly d2, d1, not a separately rounded 1-FRAC_1 duration.
        "reverse": lambda: build_two_segment_sequence(DELTA_2, DELTA_1, d2, d1, coords),
    }

    states = {}
    metas = {}
    for label, builder in builders.items():
        seq, meta = builder()
        states[label] = final_statevector_from_sequence(seq, N)
        metas[label] = meta
    return states, metas


# =============================================================================
# METRICS
# =============================================================================

def state_overlap(psi: np.ndarray, phi: np.ndarray) -> complex:
    return complex(np.vdot(normalize_state(psi), normalize_state(phi)))


def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    return float(abs(state_overlap(psi, phi)) ** 2)


def pure_trace_distance(psi: np.ndarray, phi: np.ndarray) -> float:
    return float(math.sqrt(max(0.0, 1.0 - fidelity(psi, phi))))


def fubini_study_angle(psi: np.ndarray, phi: np.ndarray) -> float:
    ov = abs(state_overlap(psi, phi))
    ov = min(1.0, max(0.0, ov))
    return float(math.acos(ov))


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    p = normalize_prob(p)
    q = normalize_prob(q)
    return float(0.5 * np.sum(np.abs(p - q)))


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    p = normalize_prob(p)
    q = normalize_prob(q)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = normalize_prob(p)
    q = normalize_prob(q)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def bhattacharyya_coeff(p: np.ndarray, q: np.ndarray) -> float:
    p = normalize_prob(p)
    q = normalize_prob(q)
    return float(np.sum(np.sqrt(p * q)))


def phase_gap(psi: np.ndarray, phi: np.ndarray) -> tuple[float, float, float]:
    p = probs_from_state(psi)
    q = probs_from_state(phi)
    bc = bhattacharyya_coeff(p, q)
    ov_abs = abs(state_overlap(psi, phi))
    return float(bc - ov_abs), float(bc), float(ov_abs)


def phase_aligned_l2(psi: np.ndarray, phi: np.ndarray) -> float:
    psi = normalize_state(psi)
    phi = normalize_state(phi)
    ov = state_overlap(psi, phi)
    if abs(ov) > 1e-15:
        phi = np.exp(-1j * np.angle(ov)) * phi
    return float(np.linalg.norm(psi - phi))


def pair_metrics(name_a: str, psi_a: np.ndarray, name_b: str, psi_b: np.ndarray) -> dict:
    p = probs_from_state(psi_a)
    q = probs_from_state(psi_b)
    gap, bc, ov_abs = phase_gap(psi_a, psi_b)

    return {
        "pair": f"{name_a}_vs_{name_b}",
        "fidelity": fidelity(psi_a, psi_b),
        "overlap_abs": ov_abs,
        "pure_trace_distance": pure_trace_distance(psi_a, psi_b),
        "fubini_study_angle_rad": fubini_study_angle(psi_a, psi_b),
        "phase_aligned_l2": phase_aligned_l2(psi_a, psi_b),
        "classical_BC": bc,
        "phase_gap_BC_minus_overlap": gap,
        "TVD_distribution": tvd(p, q),
        "Hellinger_distribution": hellinger(p, q),
        "JSD_distribution": jsd(p, q),
    }


def schedule_metrics(label: str, psi: np.ndarray, meta: dict) -> dict:
    p = probs_from_state(psi)
    labels = state_labels(N)
    top = sorted(zip(labels, p), key=lambda x: -x[1])

    row = {"schedule": label, **meta}
    for bit, prob in zip(labels, p):
        row[f"P_{bit}"] = float(prob)
    row["top_state"] = str((top[0][0], float(top[0][1])))
    return row


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    t0 = time.time()

    header("N=2 PULSER PATH-ORDERING MEMORY DIAGNOSTIC — CLOCK4")

    raw_total = raw_duration_from_loop_ns()
    d1, d2, total = forward_segment_durations()
    avg_det = weighted_avg_detuning(DELTA_1, DELTA_2, d1, d2)

    print("N:", N)
    print("spacing_um:", SPACING_UM)
    print("Omega:", OMEGA)
    print("clock_ns:", CLOCK_NS)
    print("raw duration total ns:", raw_total)
    print("clock-aligned duration total ns:", total)
    print("forward:", DELTA_1, "for", d1, "ns ->", DELTA_2, "for", d2, "ns")
    print("reverse:", DELTA_2, "for", d2, "ns ->", DELTA_1, "for", d1, "ns")
    print("weighted avg detuning, clock-aligned:", avg_det)
    print("base detuning:", BASE_DETUNING)
    print("pulse area proxy Omega*T_us:", OMEGA * total / 1000.0)

    states, metas = build_all_schedules()

    schedule_rows = [
        schedule_metrics(label, states[label], metas[label])
        for label in ["base", "avg", "forward", "reverse"]
    ]

    pair_list = [
        ("forward", "reverse"),
        ("forward", "avg"),
        ("reverse", "avg"),
        ("forward", "base"),
        ("reverse", "base"),
        ("avg", "base"),
    ]
    pair_rows = [pair_metrics(a, states[a], b, states[b]) for a, b in pair_list]

    sched_df = pd.DataFrame(schedule_rows)
    pair_df = pd.DataFrame(pair_rows)

    sched_df.to_csv(SCHEDULE_CSV, index=False)
    pair_df.to_csv(PAIR_CSV, index=False)

    header("PAIR METRICS")
    show_cols = [
        "pair",
        "fidelity",
        "pure_trace_distance",
        "fubini_study_angle_rad",
        "TVD_distribution",
        "phase_gap_BC_minus_overlap",
        "classical_BC",
        "overlap_abs",
    ]
    print(pair_df[show_cols].to_string(index=False))

    header("INTERPRETATION")
    core = pair_df[pair_df["pair"] == "forward_vs_reverse"].iloc[0]
    print("forward_vs_reverse")
    print("  Same total duration:", metas["forward"]["duration_total_ns"] == metas["reverse"]["duration_total_ns"])
    print("  Same pulse area proxy:", abs(metas["forward"]["pulse_area_proxy"] - metas["reverse"]["pulse_area_proxy"]) < 1e-12)
    print("  Same weighted avg detuning:", abs(metas["forward"]["avg_detuning"] - metas["reverse"]["avg_detuning"]) < 1e-12)
    print("  Forward avg detuning:", metas["forward"]["avg_detuning"])
    print("  Reverse avg detuning:", metas["reverse"]["avg_detuning"])
    print("  Fidelity:", core["fidelity"])
    print("  Pure-state trace distance:", core["pure_trace_distance"])
    print("  Distribution TVD:", core["TVD_distribution"])
    print("  Phase gap:", core["phase_gap_BC_minus_overlap"])

    print("  PASS_STATE_DISTANCE_GT_0p05:", bool(core["pure_trace_distance"] > 0.05))
    print("  PASS_PHASE_GAP_GT_0p001:", bool(core["phase_gap_BC_minus_overlap"] > 0.001))

    labels = state_labels(N)
    x = np.arange(len(labels))
    width = 0.20

    plt.figure(figsize=(8, 4.8))
    for k, label in enumerate(["base", "avg", "forward", "reverse"]):
        p = probs_from_state(states[label])
        plt.bar(x + (k - 1.5) * width, p, width=width, label=label)
    plt.xticks(x, labels)
    plt.ylabel("Probability")
    plt.title("N=2 output distributions: path-ordering memory, 4 ns clock-aligned")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=180)
    plt.show()

    cert = {
        "experiment_name": "N=2 Pulser path-ordering memory diagnostic, 4 ns clock-aligned",
        "N": N,
        "spacing_um": SPACING_UM,
        "Omega": OMEGA,
        "clock_ns": CLOCK_NS,
        "raw_duration_total_ns": raw_total,
        "duration_total_ns": total,
        "forward": {
            "detuning_1": DELTA_1,
            "detuning_2": DELTA_2,
            "duration_1_ns": d1,
            "duration_2_ns": d2,
            "frac_1_actual": d1 / total,
            "avg_detuning": metas["forward"]["avg_detuning"],
            "pulse_area_proxy": metas["forward"]["pulse_area_proxy"],
        },
        "reverse": {
            "detuning_1": DELTA_2,
            "detuning_2": DELTA_1,
            "duration_1_ns": d2,
            "duration_2_ns": d1,
            "frac_1_actual": d2 / total,
            "avg_detuning": metas["reverse"]["avg_detuning"],
            "pulse_area_proxy": metas["reverse"]["pulse_area_proxy"],
        },
        "important_scope": (
            "Local Pulser/Qutip exact-state simulation only. Designed as a two-qubit minimal adaptation "
            "for Hamiltonian-learning diagnostics. All pulse durations are pre-quantized to the Pulser "
            "DigitalAnalogDevice 4 ns channel clock, so printed control parameters equal executed controls. "
            "It tests whether forward/reverse piecewise schedules with equal duration, pulse area, and average "
            "detuning are distinguishable at statevector and probability levels."
        ),
        "pair_metrics": pair_df.to_dict(orient="records"),
        "schedule_metrics": sched_df.to_dict(orient="records"),
    }

    with open(CERT_JSON, "w") as f:
        json.dump(cert, f, indent=2)

    header("SAVED")
    print("saved:", PAIR_CSV)
    print("saved:", SCHEDULE_CSV)
    print("saved:", CERT_JSON)
    print("saved:", PLOT_PATH)
    print("elapsed_sec:", time.time() - t0)

    header("DONE")


if __name__ == "__main__":
    main()
