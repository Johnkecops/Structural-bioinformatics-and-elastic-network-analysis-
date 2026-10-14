#!/usr/bin/env python3
"""
Module: 01_anm_palb2
Purpose: Site-perturbed Anisotropic Network Model (ANM) of the PALB2 WD40 domain
         (residues 853-1186) for the missense variants I887S, R1117K, I1180M.
Author: Dr. Arli Aditya Parikesit
Date: 2026
Parameters:
    - cutoff: 15.0 A Calpha-Calpha cutoff, gamma 1.0 (ProDy-style ANM defaults)
    - n_modes: 20 slowest non-trivial modes
    - f (spring factor): ratio of side-chain heavy atoms, mutant/wild type,
      applied to all springs of the mutated residue (site perturbation)
    - n_perm: random-site null permutations per variant
References:
    - Atilgan et al. 2001, Biophys. J. 80, 505-515 (ANM)
    - Oliver et al. 2009, EMBO Rep. 10, 990-996 (PDB 3EU7)
Implementation note: ProDy has no wheel for this machine, so the ANM is a plain
numpy/scipy implementation (Hessian of a Calpha elastic network). It is checked
against its defining property (six zero modes) and against crystal B-factors.
"""
import json, sys
import numpy as np
from scipy.linalg import eigh
from scipy.stats import pearsonr

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
CUT, N_MODES, N_PERM, SEED = 15.0, 20, 500, 42
DOMAIN = (853, 1186)
# (name, residue, wt, mut, wt side-chain heavy atoms, mut side-chain heavy atoms)
VARIANTS = [("I887S", 887, "ILE", "SER", 4, 2),
            ("R1117K", 1117, "ARG", "LYS", 7, 5),
            ("I1180M", 1180, "ILE", "MET", 4, 4)]


def read_ca(path, chain, lo=None, hi=None):
    out = []
    for l in open(path):
        if l.startswith("ATOM") and l[12:16] == " CA " and l[21] == chain and l[16] in " A":
            r = int(l[22:26])
            if (lo is None or r >= lo) and (hi is None or r <= hi):
                out.append((r, l[17:20], np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]),
                            float(l[60:66])))
    return out


def hessian(xyz, scale=None):
    """ANM Hessian; scale is an (n,) array of per-residue spring factors (pair factor = min)."""
    n = len(xyz)
    d = xyz[:, None, :] - xyz[None, :, :]
    r2 = (d ** 2).sum(-1)
    s = np.ones(n) if scale is None else scale
    g = np.minimum(s[:, None], s[None, :]) * ((r2 <= CUT ** 2) & (r2 > 0))
    blk = -g[:, :, None, None] * d[:, :, :, None] * d[:, :, None, :] / np.where(r2 > 0, r2, 1)[:, :, None, None]
    H = blk.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n).copy()
    diag = -blk.sum(1)                      # (n,3,3)
    for i in range(n):
        H[3*i:3*i+3, 3*i:3*i+3] = diag[i]
    return H


def modes(H, k=N_MODES):
    w, v = eigh(H, subset_by_index=[0, k + 5])
    return w[6:], v[:, 6:], w[:6]


def msf_corr(w, v, n):
    inv = 1.0 / w
    uk = v.reshape(n, 3, -1)
    msf = (inv * (uk ** 2).sum(1)).sum(1)
    # cross-correlation C_ij = sum_k (1/w_k) u_i,k . u_j,k
    C = np.einsum("k,ixk,jxk->ij", inv, uk, uk)
    dg = np.sqrt(np.diag(C))
    return msf, C / np.outer(dg, dg)


def subspace_overlap(v0, v1, m=5):
    a, b = v0[:, :m], v1[:, :m]
    return float(np.sqrt(((a.T @ b) ** 2).sum() / m))


def metrics(wt, scale, idx_site, iface, xyz, msf0, C0, v0):
    n = len(xyz)
    w, v, _ = modes(hessian(xyz, scale))
    msf, C = msf_corr(w, v, n)
    rel = (msf - msf0) / msf0
    return dict(global_abs_dmsf_pct=100 * np.abs(rel).mean(),
                site_dmsf_pct=100 * rel[idx_site],
                coupling_site_iface=C[idx_site, iface].mean(),
                coupling_d=C[idx_site, iface].mean() - C0[idx_site, iface].mean(),
                overlap_5=subspace_overlap(v0, v), rel=rel)


def main():
    af = [a for a in read_ca(f"{ROOT}/data/raw/AF-Q86YC2-F1-model_v6.pdb", "A", *DOMAIN)]
    res = np.array([a[0] for a in af]); xyz = np.array([a[2] for a in af]); plddt = np.array([a[3] for a in af])
    n = len(af)
    assert n == DOMAIN[1] - DOMAIN[0] + 1, n
    names = {a[0]: a[1] for a in af}
    for _, r, wt, *_ in VARIANTS:
        assert names[r] == wt, (r, names[r])

    # Interface from PDB 3EU7: PALB2 residues with any heavy atom within 4.5 A of BRCA2 peptide (chain X)
    A, X = [], []
    for l in open(f"{ROOT}/data/raw/3EU7.pdb"):
        if l.startswith("ATOM"):
            rec = (int(l[22:26]), np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]))
            (A if l[21] == "A" else X if l[21] == "X" else []).append(rec)
    xa = np.array([a[1] for a in A]); xx = np.array([a[1] for a in X])
    dmin = np.sqrt(((xa[:, None] - xx[None]) ** 2).sum(-1)).min(1)
    iface_res = sorted({A[i][0] for i in np.where(dmin <= 4.5)[0]} & set(res.tolist()))
    iface = np.array([int(np.where(res == r)[0][0]) for r in iface_res])

    # WT ANM and validation
    H0 = hessian(xyz)
    w_all = eigh(H0, eigvals_only=True, subset_by_index=[0, 7])
    assert (np.abs(w_all[:6]) < 1e-8).all() and w_all[6] > 1e-3, w_all   # six trivial modes
    w0, v0, _ = modes(H0)
    msf0, C0 = msf_corr(w0, v0, n)

    xt = read_ca(f"{ROOT}/data/raw/3EU7.pdb", "A", *DOMAIN)
    rx = np.array([a[0] for a in xt]); bx = np.array([a[3] for a in xt]); cx = np.array([a[2] for a in xt])
    wx, vx, _ = modes(hessian(cx)); mx, _ = msf_corr(wx, vx, len(cx))
    r_b, p_b = pearsonr(mx, bx)
    r_af, _ = pearsonr(msf0[np.isin(res, rx)], mx[np.isin(rx, res)])

    out = dict(n_calpha=n, cutoff=CUT, n_modes=N_MODES, mean_plddt=float(plddt.mean()),
               min_plddt=float(plddt.min()), n_plddt_lt70=int((plddt < 70).sum()),
               n_iface=len(iface_res), iface_res=iface_res,
               first_eigvals=[float(x) for x in w0[:5]],
               valid_pearson_msf_vs_Bfactor_3EU7=float(r_b), valid_n=len(rx),
               msf_AF_vs_3EU7_pearson=float(r_af),
               wt_site_msf_percentile={}, iface_mean_msf_rel=float(msf0[iface].mean() / msf0.mean()),
               variants=[], hinge_res=[int(res[i]) for i in np.argsort(msf0)[:5]])
    rng = np.random.default_rng(SEED)
    cand = np.setdiff1d(np.arange(n), iface)
    rows, curves = [], {}
    for name, r, wt, mut, nwt, nmut in VARIANTS:
        i = int(np.where(res == r)[0][0])
        f = nmut / nwt
        out["wt_site_msf_percentile"][name] = float(100 * (msf0 < msf0[i]).mean())
        site_nb = int(((xyz - xyz[i]) ** 2).sum(1).__le__(CUT ** 2).sum() - 1)
        entry = dict(variant=name, residue=r, factor=f, n_neighbours_15A=site_nb,
                     wt_msf_rel=float(msf0[i] / msf0.mean()),
                     d_to_iface_min=float(np.sqrt(((xyz[iface] - xyz[i]) ** 2).sum(1)).min()),
                     sensitivity={})
        if f < 1.0:
            for fac in (0.8, 1.0, 1.2):
                ff = min(1.0, f * fac)
                s = np.ones(n); s[i] = ff
                m = metrics(None, s, i, iface, xyz, msf0, C0, v0)
                entry["sensitivity"][str(fac)] = {k: float(v) for k, v in m.items() if k != "rel"}
            s = np.ones(n); s[i] = f
            m = metrics(None, s, i, iface, xyz, msf0, C0, v0); curves[name] = m["rel"]
            # random-site null at the same factor (interface excluded from candidate sites)
            ng, nc = [], []
            for k in rng.choice(cand, N_PERM, replace=False if N_PERM <= len(cand) else True):
                s = np.ones(n); s[k] = f
                mm = metrics(None, s, k, iface, xyz, msf0, C0, v0)
                ng.append(mm["global_abs_dmsf_pct"]); nc.append(mm["coupling_d"])
            ng, nc = np.array(ng), np.array(nc)
            entry.update({k: float(v) for k, v in m.items() if k != "rel"})
            entry["null_global_mean"], entry["null_global_sd"] = float(ng.mean()), float(ng.std())
            entry["z_global"] = float((m["global_abs_dmsf_pct"] - ng.mean()) / ng.std())
            entry["null_coupling_mean"], entry["null_coupling_sd"] = float(nc.mean()), float(nc.std())
            entry["z_coupling"] = float((m["coupling_d"] - nc.mean()) / nc.std())
        else:
            entry.update(global_abs_dmsf_pct=0.0, site_dmsf_pct=0.0, coupling_d=0.0, overlap_5=1.0,
                         note="no spring change under heavy-atom-count scheme")
            curves[name] = np.zeros(n)
        out["variants"].append(entry)

    json.dump(out, open(f"{ROOT}/results/data/anm_results.json", "w"), indent=1)
    np.savez(f"{ROOT}/results/data/anm_arrays.npz", res=res, msf0=msf0, plddt=plddt, w0=w0,
             iface=iface, **{f"rel_{k}": v for k, v in curves.items()})
    print(json.dumps({k: v for k, v in out.items() if k not in ("variants",)}, indent=1))
    for e in out["variants"]:
        print(json.dumps(e, indent=1))


if __name__ == "__main__":
    main()
