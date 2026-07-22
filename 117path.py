# -*- coding: utf-8 -*-
"""
fixedH_plus_117_path_area_v3.py

Single-file extension for Y.Y.N. Li's neutral-atom pulse path-ordering work.

Adds two missing pieces, with v3 audit fixes:
A) N=2 shared fixed-H simultaneous-fit certificate.
B) 117-point statevector path-area scan, default N=8, 9 gaps x 13 fractions.

V2/V3 audit fixes:
   - prints nonzero_gap_only correlations alongside all_117 correlations.
   - quotes rho by witness; TVD is not folded into rho>0.97 headline.
   - explicitly marks N=2 and 117 scan as sister experiments.
   - recomputes matching flags from metadata and labels analytic fixed-H assertions.
   - v3: run command matches filename; headline citation note is generated dynamically.

Backend: local Pulser/Qutip exact-state simulation only.
Not PASQAL QPU. Not tomography. Not direct detG signature switching.

Colab install:
    !pip install -q -U pulser==1.8.0 pulser-simulation==1.8.0 pandas numpy matplotlib scipy
Run:
    python fixedH_plus_117_path_area_v3.py
"""

import math
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
except Exception:
    stats = None

from pulser import Pulse, Sequence, Register
from pulser.devices import DigitalAnalogDevice
from pulser.waveforms import ConstantWaveform
from pulser_simulation import QutipEmulator


# ============================================================
# CONFIG
# ============================================================

OUTDIR = Path("fixedH_plus_117_path_area_v3")
OUTDIR.mkdir(exist_ok=True)

RUN_N2_FIXEDH = True
RUN_117_SCAN = True

RNG_SEED = 20260722
PERM_N = 1000

SPACING_UM = 8.0
OMEGA = 1.22
CLOCK_NS = 4

# Matches your CLOCK4 N=2 log.
N2 = 2
N2_D1_NS = 2832
N2_D2_NS = 5256
N2_DELTA1 = -0.38
N2_DELTA2 = -0.25
N2_BASE_DELTA = -0.31

# Parent-paper scan defaults.
SCAN_N = 8
BASE_DETUNING = -0.31
AVG_DETUNING = -0.31
TOTAL_LOOP = 2.22
MIN_DURATION_NS = 16
MAX_DURATION_NS = 10000

# 9 x 13 = 117 points. Zero gap rows are internal null controls.
SCAN_GAPS = np.linspace(0.00, 0.24, 9)
SCAN_FRACS = np.linspace(0.10, 0.90, 13)

SAVE_STATES_N2 = True
SAVE_STATES_SCAN = False


# ============================================================
# PRINTING
# ============================================================

def header(s):
    print()
    print("=" * 100)
    print(str(s))
    print("=" * 100)


# ============================================================
# BASIC STATE / PROBABILITY HELPERS
# ============================================================

def normalize_state(psi):
    psi = np.asarray(psi, dtype=np.complex128).ravel()
    nrm = np.linalg.norm(psi)
    if nrm <= 0:
        raise ValueError("Zero statevector norm.")
    return psi / nrm


def normalize_prob(p):
    p = np.asarray(p, dtype=float).ravel()
    s = np.sum(p)
    if s <= 0:
        raise ValueError("Non-positive probability sum.")
    return p / s


def probs_from_state(psi):
    return normalize_prob(np.abs(normalize_state(psi)) ** 2)


def state_overlap(psi, phi):
    return complex(np.vdot(normalize_state(psi), normalize_state(phi)))


def state_fidelity(psi, phi):
    return float(abs(state_overlap(psi, phi)) ** 2)


def pure_trace_distance(psi, phi):
    return float(math.sqrt(max(0.0, 1.0 - state_fidelity(psi, phi))))


def fubini_study_angle(psi, phi):
    s = abs(state_overlap(psi, phi))
    return float(math.acos(max(0.0, min(1.0, s))))


def tvd_prob(p, q):
    p = normalize_prob(p)
    q = normalize_prob(q)
    return float(0.5 * np.sum(np.abs(p - q)))


def bhattacharyya_coeff(p, q):
    p = normalize_prob(p)
    q = normalize_prob(q)
    return float(np.sum(np.sqrt(p * q)))


def phase_gap_metric(psi, phi):
    p = probs_from_state(psi)
    q = probs_from_state(phi)
    bc = bhattacharyya_coeff(p, q)
    overlap_abs = abs(state_overlap(psi, phi))
    return float(bc - overlap_abs), float(bc), float(overlap_abs)


def pair_metrics(psi_a, psi_b, pair_label):
    p = probs_from_state(psi_a)
    q = probs_from_state(psi_b)
    phase_gap, bc, overlap_abs = phase_gap_metric(psi_a, psi_b)
    return {
        "pair": pair_label,
        "fidelity": state_fidelity(psi_a, psi_b),
        "overlap_abs": overlap_abs,
        "pure_trace_distance": pure_trace_distance(psi_a, psi_b),
        "fubini_study_angle_rad": fubini_study_angle(psi_a, psi_b),
        "TVD_distribution": tvd_prob(p, q),
        "phase_gap_BC_minus_overlap": phase_gap,
        "classical_BC": bc,
    }


# ============================================================
# PULSER STATEVECTOR BACKEND
# ============================================================

def nominal_coords(n, spacing_um=SPACING_UM):
    return np.array([[(i - (n - 1) / 2) * spacing_um, 0.0] for i in range(n)], dtype=float)


def make_register(n, coords=None):
    if coords is None:
        coords = nominal_coords(n)
    return Register({f"q{i}": np.array(coords[i], dtype=float) for i in range(n)})


def add_constant_pulse(seq, omega, detuning, duration_ns, phase=0.0):
    duration_ns = int(duration_ns)
    if duration_ns <= 0:
        raise ValueError("duration_ns must be positive")
    omega_wf = ConstantWaveform(duration_ns, float(omega))
    det_wf = ConstantWaveform(duration_ns, float(detuning))
    seq.add(Pulse(omega_wf, det_wf, phase), "rydberg_global")


def build_sequence_explicit(n, segments, omega=OMEGA, coords=None):
    reg = make_register(n, coords=coords)
    seq = Sequence(reg, DigitalAnalogDevice)
    seq.declare_channel("rydberg_global", "rydberg_global")
    for detuning, duration_ns in segments:
        add_constant_pulse(seq, omega, detuning, int(duration_ns))
    return seq


def get_final_state_array(result):
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


def corrected_statevector(psi, n):
    """
    Same correction used in your Pulser V2.2 workflow: bit labels are flipped
    relative to sample_final_state. The same permutation is applied to all
    states, so fidelities are unchanged.
    """
    psi = normalize_state(psi)
    dim = 2 ** n
    if len(psi) != dim:
        raise ValueError(f"len(psi)={len(psi)} != 2^N={dim}")
    out = np.zeros_like(psi)
    for i in range(dim):
        b = format(i, f"0{n}b")
        fb = "".join("1" if c == "0" else "0" for c in b)
        j = int(fb, 2)
        out[j] = psi[i]
    return normalize_state(out)


def final_statevector_from_segments(n, segments, omega=OMEGA, coords=None):
    seq = build_sequence_explicit(n, segments, omega=omega, coords=coords)
    sim = QutipEmulator.from_sequence(seq)
    result = sim.run()
    arr = get_final_state_array(result)
    arr = normalize_state(arr)
    if len(arr) != 2 ** n:
        raise RuntimeError(f"Final state dimension {len(arr)} != 2^N={2**n}")
    return corrected_statevector(arr, n)


def duration_from_loop_ns(n, omega=OMEGA, loop=TOTAL_LOOP):
    t_us = 2 * math.pi * loop / (math.sqrt(n) * omega)
    t_ns = int(round(1000 * t_us))
    t_ns = max(MIN_DURATION_NS, min(MAX_DURATION_NS, t_ns))
    t_ns = int(round(t_ns / CLOCK_NS) * CLOCK_NS)
    return max(CLOCK_NS, t_ns)


def split_duration_clock(total_ns, frac):
    total_ns = int(round(total_ns / CLOCK_NS) * CLOCK_NS)
    d1 = int(round(total_ns * frac / CLOCK_NS) * CLOCK_NS)
    d1 = max(CLOCK_NS, min(total_ns - CLOCK_NS, d1))
    d2 = total_ns - d1
    return int(d1), int(d2), int(total_ns)


def weighted_avg(det1, det2, d1_ns, d2_ns):
    return float((det1 * d1_ns + det2 * d2_ns) / (d1_ns + d2_ns))


# ============================================================
# BCH PATH-AREA PROXY
# ============================================================

def commutator_norm_proxy(n):
    """
    For H(Delta)=H_X + Delta*N + H_int, the detuning-order term is [H_X,N].
    We build H_X and N only. Interaction cancels from the simple detuning commutator
    used as the path-area proxy in this scan.
    """
    dim = 2 ** n
    HX = np.zeros((dim, dim), dtype=np.complex128)
    Nop = np.zeros((dim, dim), dtype=np.complex128)

    for x in range(dim):
        weight = bin(x).count("1")
        Nop[x, x] = weight
        for q in range(n):
            y = x ^ (1 << q)
            HX[y, x] += OMEGA / 2.0

    comm = HX @ Nop - Nop @ HX
    return float(np.linalg.norm(comm, ord=2))


# ============================================================
# N=2 SHARED FIXED-H TEST
# ============================================================

def best_shared_output_certificate(psi_f, psi_r):
    """
    Any ONE shared time-independent H acting on the same initial state at the
    same total time produces ONE output |phi>. The best possible shared fixed-H
    simultaneous fit to two pure targets is therefore the best single-state fit.

    Let s=|<psi_f|psi_r>|. Then
        max_phi ( |<psi_f|phi>|^2 + |<psi_r|phi>|^2 ) = 1+s
    and minimum total infidelity loss is
        L_min = 1-s.

    This is an optimizer-independent lower bound. Separate H_f/H_r can fit each
    target in principle; the tested restriction is one shared fixed H.
    """
    psi_f = normalize_state(psi_f)
    psi_r = normalize_state(psi_r)
    ov = state_overlap(psi_f, psi_r)
    s = abs(ov)

    if s > 1 - 1e-14:
        phi_best = psi_f.copy()
    else:
        psi_r_aligned = np.exp(-1j * np.angle(ov)) * psi_r
        phi_best = normalize_state(psi_f + psi_r_aligned)

    return {
        "target_overlap_abs": float(s),
        "target_fidelity": float(s ** 2),
        "target_pure_trace_distance": float(math.sqrt(max(0.0, 1.0 - s ** 2))),
        "shared_best_total_infidelity_loss_analytic": float(1.0 - s),
        "shared_best_per_target_infidelity_loss_analytic": float((1.0 - s) / 2.0),
        "shared_best_fidelity_each_analytic": float((1.0 + s) / 2.0),
        "shared_best_fidelity_to_forward_numeric": state_fidelity(psi_f, phi_best),
        "shared_best_fidelity_to_reverse_numeric": state_fidelity(psi_r, phi_best),
        # Analytic assertion, not a numerical optimizer result: if separate schedule-dependent
        # generators H_f and H_r are allowed, each target can be matched in principle.
        # The restriction being tested is ONE shared fixed H for both schedules.
        "separate_schedule_dependent_fixedH_lower_bound_total_loss_analytic": 0.0,
        "separate_schedule_dependent_fixedH_note": (
            "Analytic assertion, not an optimizer result: allowing separate H_f/H_r removes "
            "the shared-output constraint. The tested restriction is one shared time-independent H."
        ),
        "shared_fixedH_note": (
            "Optimizer-independent certificate: a single shared time-independent H with the same "
            "initial state and same T has one output state. It cannot exactly fit two non-identical targets."
        ),
        "interpretation": (
            "A single shared time-independent H with the same initial state and same T "
            "has one output state. It cannot exactly fit two non-identical targets. "
            "The nonzero residual 1-|<psi_f|psi_r>| is the optimizer-independent lower bound."
        ),
    }


def run_n2_fixedH():
    header("A) N=2 CLOCK4 TARGETS + SHARED FIXED-H CERTIFICATE")

    T_ns = N2_D1_NS + N2_D2_NS
    avg_det = weighted_avg(N2_DELTA1, N2_DELTA2, N2_D1_NS, N2_D2_NS)
    pulse_area_proxy = OMEGA * (T_ns / 1000.0)

    forward_segments = [(N2_DELTA1, N2_D1_NS), (N2_DELTA2, N2_D2_NS)]
    reverse_segments = [(N2_DELTA2, N2_D2_NS), (N2_DELTA1, N2_D1_NS)]
    avg_segments = [(avg_det, T_ns)]
    base_segments = [(N2_BASE_DELTA, T_ns)]

    print("N:", N2)
    print("spacing_um:", SPACING_UM)
    print("Omega:", OMEGA)
    print("T_ns:", T_ns)
    print("forward:", forward_segments)
    print("reverse:", reverse_segments)
    print("weighted avg detuning:", avg_det)
    print("pulse area proxy Omega*T_us:", pulse_area_proxy)

    psi_f = final_statevector_from_segments(N2, forward_segments)
    psi_r = final_statevector_from_segments(N2, reverse_segments)
    psi_avg = final_statevector_from_segments(N2, avg_segments)
    psi_base = final_statevector_from_segments(N2, base_segments)

    states = {"forward": psi_f, "reverse": psi_r, "avg": psi_avg, "base": psi_base}

    pair_rows = []
    for a, b in [("forward", "reverse"), ("forward", "avg"), ("reverse", "avg"),
                 ("forward", "base"), ("reverse", "base"), ("avg", "base")]:
        pair_rows.append(pair_metrics(states[a], states[b], f"{a}_vs_{b}"))

    pair_df = pd.DataFrame(pair_rows)
    print()
    print(pair_df.to_string(index=False))

    # Recompute all matching flags from metadata instead of hard-coding True.
    forward_total_ns = int(sum(d for _, d in forward_segments))
    reverse_total_ns = int(sum(d for _, d in reverse_segments))
    forward_pulse_area_proxy = float(OMEGA * (forward_total_ns / 1000.0))
    reverse_pulse_area_proxy = float(OMEGA * (reverse_total_ns / 1000.0))
    forward_avg_detuning = weighted_avg(forward_segments[0][0], forward_segments[1][0],
                                        forward_segments[0][1], forward_segments[1][1])
    reverse_avg_detuning = weighted_avg(reverse_segments[0][0], reverse_segments[1][0],
                                        reverse_segments[0][1], reverse_segments[1][1])

    cert = best_shared_output_certificate(psi_f, psi_r)
    cert.update({
        "N": N2,
        "T_ns": T_ns,
        "T_us": T_ns / 1000.0,
        "Omega": OMEGA,
        "pulse_area_proxy": pulse_area_proxy,
        "forward_segments": forward_segments,
        "reverse_segments": reverse_segments,
        "forward_total_duration_ns": forward_total_ns,
        "reverse_total_duration_ns": reverse_total_ns,
        "forward_pulse_area_proxy": forward_pulse_area_proxy,
        "reverse_pulse_area_proxy": reverse_pulse_area_proxy,
        "forward_weighted_avg_detuning": forward_avg_detuning,
        "reverse_weighted_avg_detuning": reverse_avg_detuning,
        "same_total_duration": bool(forward_total_ns == reverse_total_ns),
        "same_pulse_area_proxy": bool(np.isclose(forward_pulse_area_proxy, reverse_pulse_area_proxy, rtol=0.0, atol=1e-12)),
        "same_weighted_avg_detuning": bool(np.isclose(forward_avg_detuning, reverse_avg_detuning, rtol=0.0, atol=1e-12)),
        "weighted_avg_detuning": avg_det,
        "n2_vs_117_relation_note": (
            "The N=2 CLOCK4 target is a minimal Hamiltonian-learning interface. "
            "It is a sister experiment to the N=8 117-point scan, not a point contained inside that scan grid."
        ),
    })

    print()
    print("SHARED FIXED-H CERTIFICATE")
    for k, v in cert.items():
        print(f"  {k}: {v}")

    pair_csv = OUTDIR / "n2_pair_metrics_clock4_fixedH.csv"
    cert_json = OUTDIR / "n2_fixedH_certificate.json"
    pair_df.to_csv(pair_csv, index=False)
    with open(cert_json, "w") as f:
        json.dump(cert, f, indent=2)

    if SAVE_STATES_N2:
        np.savez_compressed(OUTDIR / "n2_target_states_clock4.npz",
                            psi_forward=psi_f, psi_reverse=psi_r, psi_avg=psi_avg, psi_base=psi_base)

    print("saved:", pair_csv)
    print("saved:", cert_json)
    if SAVE_STATES_N2:
        print("saved:", OUTDIR / "n2_target_states_clock4.npz")

    return states, cert, pair_df


# ============================================================
# 117-POINT SCAN
# ============================================================

def spearman_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if stats is not None:
        rho, p = stats.spearmanr(x, y)
        return float(rho), float(p)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rho = np.corrcoef(rx, ry)[0, 1]
    return float(rho), float("nan")


def pearson_r2(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) <= 1e-15 or np.std(y) <= 1e-15:
        return float("nan")
    r = np.corrcoef(x, y)[0, 1]
    return float(r ** 2)


def permutation_p_spearman(x, y, observed_rho, n_perm=PERM_N, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    count = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        rho, _ = spearman_corr(x, yp)
        if rho >= observed_rho:
            count += 1
    return float((count + 1) / (n_perm + 1))


def fit_summary_for_witness(df, xcol, ycol):
    x = df[xcol].astype(float).values
    y = df[ycol].astype(float).values
    rho, scipy_p = spearman_corr(x, y)
    r2 = pearson_r2(x, y)
    perm_p = permutation_p_spearman(x, y, rho)
    return {
        "x": xcol,
        "y": ycol,
        "n": int(len(df)),
        "spearman_rho": rho,
        "spearman_p_scipy": scipy_p,
        "pearson_R2": r2,
        "permutation_p_one_sided": perm_p,
    }


def _rho_lookup(summary_df, subset, y, x="C_nc_simple"):
    rows = summary_df[
        (summary_df["subset"] == subset)
        & (summary_df["x"] == x)
        & (summary_df["y"] == y)
    ]
    if len(rows) != 1:
        return None
    return float(rows.iloc[0]["spearman_rho"])


def make_headline_citation_note(summary_df):
    """Generate the headline/citation note from actual results."""
    labels = {
        "pure_trace_distance": "D_pure",
        "phase_gap_BC_minus_overlap": "Gamma_coh",
        "TVD_distribution": "TVD",
    }
    subset_parts = []
    for subset in ["all_117_including_zero_gap", "nonzero_gap_only"]:
        entries = []
        for y, label in labels.items():
            rho = _rho_lookup(summary_df, subset, y)
            entries.append(f"{label}={rho:.3f}" if rho is not None else f"{label}=NA")
        subset_parts.append(f"{subset}: " + ", ".join(entries))

    all_rhos = {
        label: _rho_lookup(summary_df, "all_117_including_zero_gap", y)
        for y, label in labels.items()
    }
    above_097 = [label for label, rho in all_rhos.items() if rho is not None and rho > 0.97]
    not_above_097 = [label for label, rho in all_rhos.items() if rho is not None and rho <= 0.97]

    if not_above_097:
        headline_rule = (
            "Do not state that all witnesses have rho > 0.97. "
            f"In this run, rho > 0.97 holds for {', '.join(above_097) or 'none'}, "
            f"but not for {', '.join(not_above_097)}."
        )
    else:
        headline_rule = "In this run, rho > 0.97 holds for all listed witnesses."

    return "Quote correlations by witness. " + " | ".join(subset_parts) + ". " + headline_rule


def run_117_scan():
    header("B) 117-POINT STATEVECTOR PATH-AREA SCAN")

    n = SCAN_N
    coords = nominal_coords(n)
    total_ns = duration_from_loop_ns(n)
    comm_norm = commutator_norm_proxy(n)

    print("N:", n)
    print("total_ns:", total_ns)
    print("T_us:", total_ns / 1000.0)
    print("Omega:", OMEGA)
    print("avg detuning target:", AVG_DETUNING)
    print("points:", len(SCAN_GAPS) * len(SCAN_FRACS))
    print("||[H_X,N]|| proxy:", comm_norm)

    rows = []
    saved_states = {}
    idx = 0
    t0 = time.time()

    for gap in SCAN_GAPS:
        for frac_nom in SCAN_FRACS:
            idx += 1
            d1_ns, d2_ns, T_ns = split_duration_clock(total_ns, frac_nom)
            f_actual = d1_ns / T_ns

            # Weighted average constraint and gap constraint:
            # f*Delta1 + (1-f)*Delta2 = AVG_DETUNING
            # Delta2 - Delta1 = gap
            delta1 = AVG_DETUNING - (1.0 - f_actual) * gap
            delta2 = AVG_DETUNING + f_actual * gap
            avg_actual = weighted_avg(delta1, delta2, d1_ns, d2_ns)

            forward_segments = [(delta1, d1_ns), (delta2, d2_ns)]
            reverse_segments = [(delta2, d2_ns), (delta1, d1_ns)]

            psi_f = final_statevector_from_segments(n, forward_segments, coords=coords)
            psi_r = final_statevector_from_segments(n, reverse_segments, coords=coords)

            m = pair_metrics(psi_f, psi_r, "forward_vs_reverse")
            Cnc = abs(delta2 - delta1) * (d1_ns / 1000.0) * (d2_ns / 1000.0) * comm_norm
            Cnc_simple = abs(delta2 - delta1) * f_actual * (1.0 - f_actual)

            # Recompute schedule-matching flags from metadata instead of hard-coding.
            forward_total_ns = int(sum(d for _, d in forward_segments))
            reverse_total_ns = int(sum(d for _, d in reverse_segments))
            forward_area = float(OMEGA * (forward_total_ns / 1000.0))
            reverse_area = float(OMEGA * (reverse_total_ns / 1000.0))
            forward_avg = weighted_avg(forward_segments[0][0], forward_segments[1][0],
                                       forward_segments[0][1], forward_segments[1][1])
            reverse_avg = weighted_avg(reverse_segments[0][0], reverse_segments[1][0],
                                       reverse_segments[0][1], reverse_segments[1][1])

            row = {
                "idx": idx,
                "N": n,
                "gap": float(gap),
                "frac_nominal": float(frac_nom),
                "frac_actual": float(f_actual),
                "delta1": float(delta1),
                "delta2": float(delta2),
                "duration1_ns": int(d1_ns),
                "duration2_ns": int(d2_ns),
                "duration_total_ns": int(T_ns),
                "T_us": T_ns / 1000.0,
                "weighted_avg_detuning": float(avg_actual),
                "forward_weighted_avg_detuning": float(forward_avg),
                "reverse_weighted_avg_detuning": float(reverse_avg),
                "avg_target": AVG_DETUNING,
                "forward_total_duration_ns": forward_total_ns,
                "reverse_total_duration_ns": reverse_total_ns,
                "forward_pulse_area_proxy": forward_area,
                "reverse_pulse_area_proxy": reverse_area,
                "same_total_duration": bool(forward_total_ns == reverse_total_ns),
                "same_pulse_area_proxy": bool(np.isclose(forward_area, reverse_area, rtol=0.0, atol=1e-12)),
                "same_weighted_avg_detuning": bool(np.isclose(forward_avg, reverse_avg, rtol=0.0, atol=1e-12)),
                "same_avg_target_after_clock_alignment": bool(np.isclose(avg_actual, AVG_DETUNING, rtol=0.0, atol=1e-12)),
                "commutator_norm_proxy": comm_norm,
                "C_nc": float(Cnc),
                "C_nc_simple": float(Cnc_simple),
            }
            row.update(m)
            rows.append(row)

            if SAVE_STATES_SCAN:
                saved_states[f"psi_f_{idx:03d}"] = psi_f
                saved_states[f"psi_r_{idx:03d}"] = psi_r

            print(f"[{idx:03d}/117] gap={gap:.4f} f={f_actual:.4f} "
                  f"D={m['pure_trace_distance']:.6f} "
                  f"Gamma={m['phase_gap_BC_minus_overlap']:.6f} "
                  f"TVD={m['TVD_distribution']:.6f} "
                  f"elapsed={time.time() - t0:.1f}s")

    df = pd.DataFrame(rows)
    scan_csv = OUTDIR / "scan117_statevector_metrics.csv"
    df.to_csv(scan_csv, index=False)

    if SAVE_STATES_SCAN:
        np.savez_compressed(OUTDIR / "scan117_target_states.npz", **saved_states)

    summaries = []
    for subset_name, sub in [("all_117_including_zero_gap", df), ("nonzero_gap_only", df[df["gap"] > 0].copy())]:
        for xcol in ["C_nc", "C_nc_simple"]:
            for ycol in ["pure_trace_distance", "phase_gap_BC_minus_overlap", "TVD_distribution", "fubini_study_angle_rad"]:
                s = fit_summary_for_witness(sub, xcol, ycol)
                s["subset"] = subset_name
                summaries.append(s)

    summary_df = pd.DataFrame(summaries)
    summary_csv = OUTDIR / "scan117_correlation_summary.csv"
    summary_json = OUTDIR / "scan117_certificate.json"
    summary_df.to_csv(summary_csv, index=False)

    cert = {
        "experiment": "117-point statevector path-area scan",
        "N": n,
        "points": int(len(df)),
        "grid": {"gaps": [float(x) for x in SCAN_GAPS], "fracs": [float(x) for x in SCAN_FRACS]},
        "zero_gap_control_points": int((df["gap"] == 0).sum()),
        "nonzero_gap_points": int((df["gap"] > 0).sum()),
        "total_duration_ns": int(total_ns),
        "Omega": OMEGA,
        "avg_detuning_target": AVG_DETUNING,
        "commutator_norm_proxy": comm_norm,
        "correlations": summary_df.to_dict(orient="records"),
        "headline_citation_note": make_headline_citation_note(summary_df),
        "n2_vs_117_relation_note": (
            "The N=2 CLOCK4 diagnostic and this N=8 117-point scan are sister experiments. "
            "The scan validates the same C_nc ∝ |Delta2-Delta1| f(1-f) law in a different N, "
            "duration, average-detuning window, and parameter grid; it does not contain the N=2 point."
        ),
        "important_limit": (
            "Local Pulser/Qutip statevector simulation only. This is not PASQAL QPU tomography. "
            "Correlation strength depends on the chosen perturbative scan window and backend details."
        ),
    }
    with open(summary_json, "w") as f:
        json.dump(cert, f, indent=2)

    for ycol, name in [("pure_trace_distance", "D_pure"),
                       ("phase_gap_BC_minus_overlap", "Gamma_coh"),
                       ("TVD_distribution", "TVD")]:
        plt.figure(figsize=(7, 5))
        plt.scatter(df["C_nc_simple"], df[ycol], s=28)
        plt.xlabel(r"$|\Delta_2-\Delta_1| f(1-f)$")
        plt.ylabel(ycol)
        plt.title(f"117-point scan: {name} vs path-area proxy")
        plt.tight_layout()
        plot_path = OUTDIR / f"scan117_{name}_vs_Cnc_simple.png"
        plt.savefig(plot_path, dpi=180)
        plt.close()

    print()
    print("CORRELATION SUMMARY — C_nc_simple")
    show = summary_df[(summary_df["x"] == "C_nc_simple")
                      & (summary_df["subset"].isin(["all_117_including_zero_gap", "nonzero_gap_only"]))
                      & (summary_df["y"].isin(["pure_trace_distance", "phase_gap_BC_minus_overlap", "TVD_distribution"]))]
    show = show[["subset", "x", "y", "n", "spearman_rho", "spearman_p_scipy", "pearson_R2", "permutation_p_one_sided"]]
    print(show.to_string(index=False))
    print()
    print("HEADLINE CITATION NOTE")
    print(make_headline_citation_note(summary_df))
    print()
    print("RELATION NOTE")
    print("  N=2 and 117-scan are sister experiments, not an inclusion relationship.")

    print("saved:", scan_csv)
    print("saved:", summary_csv)
    print("saved:", summary_json)
    print("saved plots:", OUTDIR / "scan117_*_vs_Cnc_simple.png")

    return df, summary_df, cert


# ============================================================
# MAIN
# ============================================================

def main():
    header("FIXED-H + 117-POINT PATH-AREA EXTENSION")
    print("OUTDIR:", OUTDIR)
    print("RUN_N2_FIXEDH:", RUN_N2_FIXEDH)
    print("RUN_117_SCAN:", RUN_117_SCAN)
    print()
    print("Important:")
    print("  - N=2 fixed-H part is an optimizer-independent shared-output lower bound.")
    print("  - 117 scan is local Pulser/Qutip statevector simulation, not QPU.")
    print("  - If rho differs from your paper number, use the exact paper grid/backend window.")

    t0 = time.time()
    if RUN_N2_FIXEDH:
        run_n2_fixedH()
    if RUN_117_SCAN:
        run_117_scan()
    header("DONE")
    print("elapsed_sec:", time.time() - t0)
    print("saved all outputs under:", OUTDIR)


if __name__ == "__main__":
    main()
