#!/usr/bin/env python3
"""
Module: 02_anm_figures
Purpose: Figures 6 and 7 (ANM MSF profile; site-perturbation response) from results/data.
Author: Dr. Arli Aditya Parikesit
Date: 2026
Parameters: argv[1] project root
"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = sys.argv[1] if len(sys.argv) > 1 else "."
d = np.load(f"{R}/results/data/anm_arrays.npz")
res, msf, iface = d["res"], d["msf0"], d["iface"]
sites = {"I887S": 887, "R1117K": 1117, "I1180M": 1180}
fig, ax = plt.subplots(figsize=(9, 3.6), dpi=200)
ax.plot(res, msf / msf.mean(), color="#1f4e79", lw=1.2)
ax.scatter(res[iface], (msf / msf.mean())[iface], s=14, color="#c0392b", zorder=3, label="BRCA2-contact residues (3EU7)")
for (n, r), c in zip(sites.items(), ["#e69f00", "#009e73", "#7b3294"]):
    ax.axvline(r, color=c, ls="--", lw=1.1, label=n)
ax.set_xlabel("Residue number"); ax.set_ylabel("Relative MSF (ANM, mean = 1)")
ax.legend(fontsize=7, frameon=False, ncol=2); fig.tight_layout()
fig.savefig(f"{R}/results/figures/Figure6_ANM_MSF.png"); plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 3.2), dpi=200)
for n, c in zip(["I887S", "R1117K"], ["#e69f00", "#009e73"]):
    ax.plot(res, 100 * d[f"rel_{n}"], color=c, lw=1.2, label=n)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("Residue number"); ax.set_ylabel("Change in MSF versus WT (%)")
ax.legend(fontsize=8, frameon=False); fig.tight_layout()
fig.savefig(f"{R}/results/figures/Figure7_ANM_dMSF.png"); plt.close(fig)
print("ok", (100*d["rel_I887S"]).max(), res[np.argmax(d["rel_I887S"])], (100*d["rel_R1117K"]).max())
