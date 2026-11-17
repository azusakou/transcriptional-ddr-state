#!/usr/bin/env python3
"""Render the manuscript's three vector figures from supplied analysis outputs."""
from pathlib import Path
import os, sys, argparse, tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ddr_matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
HERE = args.data
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
BLUE = "#4C78A8"
TEAL = "#398579"
ORANGE = "#D88B41"
PURPLE = "#8873A6"
GRAY = "#8A929D"
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    }
)
F = pd.read_csv(HERE / "forecast/performance.csv")
P = pd.read_csv(HERE / "forecast/nested_predictions.csv")
B = pd.read_csv(HERE / "forecast/line_bootstrap.csv")
F = F[
    (F.analysis == "composite_z2")
    & (F.cohort == "time_specific")
    & (F.split == "frozen_outer")
]
P = P[
    (P.analysis == "composite_z2")
    & (P.cohort == "time_specific")
    & (P.split == "frozen_outer")
]
B = B[
    (B.analysis == "composite_z2")
    & (B.cohort == "time_specific")
    & (B.split == "frozen_outer")
]


def panel(ax, letter, title):
    ax.set_title(title, loc="left", pad=9)
    ax.text(
        -0.14,
        1.08,
        letter,
        transform=ax.transAxes,
        weight="bold",
        fontsize=10,
        va="bottom",
    )


def save(fig, name):
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight", pad_inches=0.05)
    if os.environ.get("FIGURE_PREVIEW_DIR"):
        folder = Path(os.environ["FIGURE_PREVIEW_DIR"])
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            folder / (name + ".png"), dpi=180, bbox_inches="tight", pad_inches=0.05
        )
    plt.close(fig)


# Figure 1: the main comparison is against simpler models, with identical held-line folds.
fig, axs = plt.subplots(
    2, 2, figsize=(7.2, 5.7), gridspec_kw={"width_ratios": [1, 1.1]}
)
fig.subplots_adjust(
    hspace=0.65, wspace=0.62, left=0.12, right=0.98, bottom=0.1, top=0.94
)
ax = axs[0, 0]
names = ["Adamson", "Norman", "RPE1", "sci-Plex4", "sci-Plex2", "McFarland"]
counts = [15006, 111445, 247914, 98437, 24262, 182875]
colors = [GRAY] * 4 + [TEAL, BLUE]
y = np.arange(6)
ax.barh(y, counts, color=colors, height=0.58)
ax.set_yticks(y, names)
ax.set_xscale("log")
ax.set_xlim(8e3, 5e5)
ax.set_xlabel("Profiled cells (log scale)")
panel(ax, "a", "Inventory and predictive resources")
ax.text(
    0.98,
    0.03,
    "Colour: predictive resources\nGrey: other profiled resources",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=6,
)
ax = axs[0, 1]
models = [
    "drug_only",
    "baseline_only",
    "cycle_only_plus_drug",
    "ddr_fraction_additive",
    "no_DDR_TP53_plus_drug",
    "parsimonious_additive",
]
labels = [
    "Regimen only",
    "Baseline only",
    "Cycle + regimen",
    "DDR fraction + regimen",
    "Cycle/stress + regimen",
    "Four features + regimen",
]
q = F[F.drug == "all"].set_index("model")
for i, m in enumerate(models):
    row = q.loc[m]
    ax.scatter(
        row.r2, i, color=BLUE if m == "parsimonious_additive" else TEAL, s=26, zorder=3
    )
    z = B[
        (B.model == m)
        & (B.reference_model == "none")
        & (B.drug == "all")
        & (B.statistic == "r2")
    ].iloc[0]
    ax.plot(
        [z.ci_low, z.ci_high],
        [i, i],
        color=BLUE if m == "parsimonious_additive" else TEAL,
        lw=1.1,
    )
    ax.text(0.70, i, f"{row.r2:.3f}", ha="right", va="center", fontsize=6.5)
ax.axvline(0, color=GRAY, lw=0.6)
ax.set_yticks(range(len(models)), labels)
ax.invert_yaxis()
ax.set_xlim(-0.1, 0.74)
ax.set_xlabel("Held-line $R^2$ (95% line-bootstrap CI)")
panel(ax, "b", "Simple baseline comparators")
ax = axs[1, 0]
q = P[P.model == "parsimonious_additive"]
dcolors = {
    "Gemcitabine": ORANGE,
    "Idasanutlin": TEAL,
    "Prexasertib": PURPLE,
    "Bortezomib": BLUE,
}
for d, g in q.groupby("drug"):
    ax.scatter(
        g.observed,
        g.predicted,
        s=12,
        alpha=0.68,
        color=dcolors[d],
        label=d,
        edgecolors="none",
    )
ax.plot([0, 1], [0, 1], color=GRAY, lw=0.7)
ax.axhline(0, color=GRAY, lw=0.4)
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.03, 1.03)
ax.set_xlabel("Observed DDR-high fraction")
ax.set_ylabel("Four-feature bounded prediction")
ax.legend(loc="lower right", fontsize=5.5)
panel(ax, "c", "Vehicle-state response prediction")
ax = axs[1, 1]
t = pd.read_csv(HERE / "classifier/leave_task_out_task_metrics.csv")
base = t[t.feature_set == "QC_cellcycle"].set_index("task_name")
full = t[t.feature_set == "biology_QC_cellcycle"].set_index("task_name")
for i, k in enumerate(full.index):
    c = TEAL if "sciplex" in k else BLUE
    ax.plot([base.loc[k, "AP"], full.loc[k, "AP"]], [i, i], color=c, lw=1.3)
    ax.scatter(base.loc[k, "AP"], i, s=18, facecolors="white", edgecolors=c, lw=0.8)
    ax.scatter(full.loc[k, "AP"], i, s=23, color=c, zorder=3)
ax.set_yticks(range(len(full)), [k.split("__")[-1] for k in full.index])
ax.invert_yaxis()
ax.set_xlim(0, 0.61)
ax.set_xlabel("Held-task average precision")
ax.text(
    0.98,
    0.05,
    "Open: QC + context + cycle\nFilled: + stress proxies",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=6,
)
panel(ax, "d", "Increment beyond QC and cycle")
save(fig, "Fig1_compendium_and_prediction")

# Figure 2: grouped importance, normalized chemical contrasts and fixed-control external contrasts.
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.8))
fig.subplots_adjust(
    hspace=0.7, wspace=0.67, left=0.13, right=0.97, bottom=0.12, top=0.94
)
ax = axs[0, 0]
imp = pd.read_csv(HERE / "classifier/grouped_permutation_importance_by_task.csv")
# by-task mean across repeat permutations; show all six tasks, rather than treating repeats as experiments.
for i, (group, label) in enumerate(
    [
        ("QC_joint", "QC"),
        ("cellcycle_programmes_joint", "Cycle / E2F"),
        ("stress_programmes_joint", "Stress proxies"),
    ]
):
    g = imp[imp.group == group]
    vals = g.AP_drop.to_numpy()
    ax.scatter(
        vals,
        np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)),
        s=16,
        color=BLUE,
        alpha=0.65,
    )
    ax.scatter(
        np.mean(vals),
        i,
        s=44,
        color=TEAL,
        marker="D",
        edgecolor="white",
        lw=0.5,
        zorder=3,
    )
ax.axvline(0, color=GRAY, lw=0.6)
ax.set_yticks([0, 1, 2], ["QC", "Cycle / E2F", "Stress proxies"])
ax.set_xlabel("AP drop after joint permutation")
panel(ax, "a", "Held-task programme contribution")
ax = axs[0, 1]
chem = pd.read_csv(
    HERE / "source_context/normal_matched_chemical_cell_line_summary.csv"
)
chem = chem[chem["mode"] == "24h"]
mods = ["TP53_axis", "E2F_replication", "ATF4_integrated_stress", "XBP1_UPR"]
drugs = ["Gemcitabine", "Idasanutlin", "Prexasertib", "Bortezomib"]
mat = np.array(
    [
        [
            chem[(chem.drug == d) & (chem.module == m)].mean_delta_fraction_z2.iloc[0]
            for m in mods
        ]
        for d in drugs
    ]
)
vm = max(abs(mat.min()), abs(mat.max()))
im = ax.imshow(
    mat, cmap="BrBG", norm=TwoSlopeNorm(vcenter=0, vmin=-vm, vmax=vm), aspect="auto"
)
ax.set_yticks(range(4), drugs)
ax.set_xticks(range(4), ["TP53", "E2F", "ATF4", "XBP1/UPR"], rotation=25, ha="right")
for i in range(4):
    for j in range(4):
        ax.text(
            j,
            i,
            f"{mat[i,j]:.2f}",
            ha="center",
            va="center",
            fontsize=6,
            color="white" if abs(mat[i, j]) > 0.6 * vm else "#222",
        )
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.set_label("Change in high fraction", fontsize=6)
panel(ax, "b", "24 h chemical programme contrasts")
ax = axs[1, 0]
e = pd.read_csv(HERE / "external/external_sample_effects.csv")
q = e[(e.dataset_id == "GSE145700") & (e.module == "ddr_composite")]
for i, gen in enumerate(["WT", "ATM_KD"]):
    g = q[q.genotype == gen]
    v = g.delta_mean_score.to_numpy()
    ax.scatter(
        np.full(len(v), i) + [-0.055, 0.055], v, s=28, color=[BLUE, ORANGE][i], zorder=3
    )
    ax.plot([i - 0.13, i + 0.13], [v.mean()] * 2, color="#222", lw=1)
ax.axhline(0, color=GRAY, lw=0.6)
ax.set_xticks([0, 1], ["Parental", "ATM-KD"])
ax.set_xlim(-0.5, 1.5)
ax.set_ylabel("IR - control mean log-normalized score")
ax.set_xlabel("Two IR libraries per genotype; unpaired")
panel(ax, "c", "GSE145700: 5 Gy IR, 4 h")
ax = axs[1, 1]
q = e[(e.dataset_id == "GSE255874") & (e.module == "ddr_composite")]
order = ["HRAS_2N4N", "HRAS_Polyploid", "Pevo_2N4N", "Pevo_Polyploid"]
for j, ex in enumerate(sorted(q.experiment.unique())):
    vals = [
        q[(q.condition == c) & (q.experiment == ex)].delta_fraction_z2.iloc[0]
        for c in order
    ]
    ax.plot(vals, np.arange(4), "-o", color=[BLUE, ORANGE][j], lw=0.8, ms=3.5, label=ex)
ax.set_yticks(
    range(4), ["HRAS 2N-4N", "HRAS polyploid", "Pevo 2N-4N", "Pevo polyploid"]
)
ax.invert_yaxis()
ax.axvline(0, color=GRAY, lw=0.6)
ax.set_xlabel("DDR-high fraction change from EV 2N-4N")
ax.legend(loc="upper right", fontsize=6)
panel(ax, "d", "GSE255874: replicate-resolved shifts")
save(fig, "Fig2_regulons_and_external_validation")

# Figure 3: retain live-trace/heatmap logic and add exact-linked endpoint damage marker.
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4))
fig.subplots_adjust(
    hspace=0.62, wspace=0.42, left=0.1, right=0.98, bottom=0.11, top=0.94
)
tracks = pd.read_csv(HERE / "external/imaging_track_summaries.csv")
live = pd.read_csv(HERE / "external/imaging_per_frame.csv.gz")
links = pd.read_csv(HERE / "external/imaging_live_endpoint_links.csv")
ax = axs[0, 0]
for tid, c in zip(
    tracks.sort_values(["max_53BP1", "duration_frames"], ascending=False)
    .head(6)
    .track_id,
    [BLUE, ORANGE, TEAL, PURPLE, "#BF6D80", "#8C8551"],
):
    g = live[live.track_id == tid].sort_values("Time").set_index("Time")
    g = g.reindex(np.arange(g.index.min(), g.index.max() + 1))
    ax.plot(g.index, g["53BP1_foci__Counts"], lw=0.9, color=c)
ax.set_xlabel("Live frame")
ax.set_ylabel("53BP1 foci")
panel(ax, "a", "Representative live-cell tracks")
ax = axs[0, 1]
ids = (
    tracks.sort_values(["max_53BP1", "duration_frames"], ascending=False)
    .head(42)
    .track_id.tolist()
)
maxlen = int(tracks.duration_frames.max())
a = np.full((len(ids), maxlen), np.nan)
for i, tid in enumerate(ids):
    g = live[live.track_id == tid].sort_values("Time")
    fr = (g.Time - g.Time.min()).astype(int).to_numpy()
    a[i, fr] = g["53BP1_foci__Counts"]
cmap = plt.colormaps["magma"].copy()
cmap.set_bad("white")
im = ax.imshow(
    a, aspect="auto", cmap=cmap, vmin=0, vmax=np.nanmax(a), interpolation="none"
)
ax.set_xlabel("Frame relative to track start")
ax.set_ylabel("High-signal tracks")
ax.set_yticks([])
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
cb.set_label("53BP1 foci", fontsize=6)
panel(ax, "b", "Live 53BP1 histories")
ax = axs[1, 0]
ax.scatter(
    tracks.max_PCNA, tracks.max_53BP1, s=13, alpha=0.55, color=BLUE, edgecolors="none"
)
ax.set_xlabel("Maximum live PCNA foci")
ax.set_ylabel("Maximum live 53BP1 foci")
ax.text(
    0.97,
    0.95,
    "244 tracks\n$\\rho$ = 0.418\nCount-adjusted: 0.094",
    ha="right",
    va="top",
    transform=ax.transAxes,
    fontsize=6.5,
)
panel(ax, "c", "Observation count affects maxima")
ax = axs[1, 1]
# Jitter only for visibility of integer ties; source values and correlations are unjittered.
rng = np.random.default_rng(221)
xx = links.max_53BP1.to_numpy()
yy = links["gH2AX_foci__Counts"].to_numpy()
ax.scatter(
    xx + rng.uniform(-0.12, 0.12, len(xx)),
    yy + rng.uniform(-0.12, 0.12, len(xx)),
    s=18,
    alpha=0.65,
    color=TEAL,
    edgecolors="none",
)
ax.set_xlabel("Maximum live 53BP1 foci")
ax.set_ylabel("Endpoint $\\gamma$H2AX foci")
ax.text(
    0.97,
    0.95,
    "95 exact-linked cells\n$\\rho$ = 0.526\nCount-adjusted: 0.632",
    ha="right",
    va="top",
    transform=ax.transAxes,
    fontsize=6.5,
)
panel(ax, "d", "Live-to-endpoint damage-marker link")
save(fig, "Fig3_lineage_same_cell_evidence")

# Supplementary Figure S1: all four drug associations and biological context controls.
V = HERE / "source_context"
G = HERE / "genetic"
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.8))
fig.subplots_adjust(
    hspace=0.72, wspace=0.58, left=0.14, right=0.98, bottom=0.12, top=0.94
)
for ax, filename, letter, title in [
    (
        axs[0, 0],
        "observed_prism_associations.csv",
        "a",
        "Observed fraction and viability",
    ),
    (
        axs[0, 1],
        "frozen_forecast_PRISM_associations.csv",
        "b",
        "Held-out prediction and viability",
    ),
]:
    q = pd.read_csv(V / filename)
    q = q[q.primary == True].set_index("drug")
    for i, drug in enumerate(
        ["Gemcitabine", "Idasanutlin", "Prexasertib", "Bortezomib"]
    ):
        r = q.loc[drug]
        c = dcolors[drug]
        ax.plot([r.ci_low, r.ci_high], [i, i], color=c, lw=1.2)
        ax.scatter(r.rho, i, color=c, s=25, zorder=3)
        ax.text(
            0.98,
            i,
            f"n={int(r.n)}, q={r.q_primary_four_tests:.3f}",
            ha="right",
            va="center",
            fontsize=5.7,
        )
    ax.axvline(0, color=GRAY, lw=0.6)
    ax.set_xlim(-1, 1.02)
    ax.set_yticks(range(4), ["Gemcitabine", "Idasanutlin", "Prexasertib", "Bortezomib"])
    ax.invert_yaxis()
    ax.set_xlabel("Spearman correlation with PRISM AUC")
    panel(ax, letter, title)
ax = axs[1, 0]
q = pd.read_csv(V / "idasanutlin_TP53_context.csv")
q = q[q["mode"] == "24h"]
rng = np.random.default_rng(42)
for i, (group, label, c) in enumerate(
    [
        ("no_damaging_or_hotspot_mutation", "No damaging /\nhotspot annotation", TEAL),
        ("damaging_or_hotspot_mutation", "Damaging / hotspot\nannotation", ORANGE),
    ]
):
    vals = q[q.TP53_group == group].delta_mean_score.to_numpy()
    ax.scatter(i + rng.uniform(-0.14, 0.14, len(vals)), vals, color=c, s=19, alpha=0.75)
    ax.plot([i - 0.22, i + 0.22], [vals.mean()] * 2, color="#222", lw=1)
ax.axhline(0, color=GRAY, lw=0.6)
ax.set_xticks(
    [0, 1], ["No damaging /\nhotspot annotation", "Damaging / hotspot\nannotation"]
)
ax.set_xlim(-0.5, 1.5)
ax.set_ylabel("Idasanutlin 24 h p53-proxy score change")
panel(ax, "c", "Source TP53 mutation context")
ax = axs[1, 1]
q = pd.read_csv(G / "target_effects.csv")
q = q[q.eligible_summary == True]
groups = ["repair_checkpoint", "proteostasis", "mitosis", "ribosome"]
colors = [BLUE, TEAL, PURPLE, ORANGE]
for i, (gr, c) in enumerate(zip(groups, colors)):
    z = q[q.predefined_group == gr]
    for offset, key, marker in [
        (-0.11, "composite_fraction_delta", "o"),
        (0.11, "p53_response_fraction_delta", "^"),
    ]:
        vals = z[key].to_numpy()
        ax.scatter(
            i + offset + rng.uniform(-0.035, 0.035, len(vals)),
            vals,
            color=c,
            s=13,
            alpha=0.65,
            marker=marker,
        )
        ax.plot(
            [i + offset - 0.07, i + offset + 0.07],
            [np.median(vals)] * 2,
            color="#222",
            lw=1,
        )
ax.axhline(0, color=GRAY, lw=0.6)
ax.set_xticks(
    range(4),
    [
        "Repair / checkpoint\n8 targets",
        "Proteostasis\n3 targets",
        "Mitosis\n4 targets",
        "Ribosome\n9 targets",
    ],
    rotation=25,
    ha="right",
)
ax.set_ylabel("Target-wise high-fraction change")
ax.text(
    0.02,
    0.98,
    "Circles: composite DDR\nTriangles: p53 response",
    ha="left",
    va="top",
    transform=ax.transAxes,
    fontsize=6,
)
panel(ax, "d", "RPE1 genetic-perturbation context")
save(fig, "Supplementary_FigS1")
