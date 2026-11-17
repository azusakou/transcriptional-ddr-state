#!/usr/bin/env python3
"""Render the same-cell damage-marker and functional-repair comparisons."""
from pathlib import Path
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    damage = args.data / 'direct_damage'
    repair = args.data / 'direct_repair'
    a = pd.read_csv(damage / 'within_plate_associations.csv')
    a = a[a.score.eq('DDR')].sort_values('plate')
    b = pd.read_csv(damage / 'within_plate_DDR_label_associations.csv')
    b = b[b.label.eq('DDR_high')].sort_values('plate')
    r = pd.read_csv(repair / 'primary_results.csv')
    assert len(a) == len(b) == 6 and len(r) == 16
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8,
        'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .6, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'legend.frameon': False,
    })
    blue, orange, grey = '#3E6A8A', '#B77643', '#A0A0A0'
    fig = plt.figure(figsize=(7.2, 6.65))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.48], hspace=.64,
                          wspace=.45, left=.27, right=.97, top=.94, bottom=.10)
    ax = fig.add_subplot(gs[0, 0])
    ay = fig.add_subplot(gs[0, 1], sharey=ax)
    az = fig.add_subplot(gs[1, :])
    y = np.arange(6)
    labels = [f"{row.plate[-3:]}  (n={row.n})" for row in a.itertuples()]
    ax.hlines(y, a.rho_adjusted_cycle, a.rho_adjusted, color=grey, lw=1)
    ax.scatter(a.rho_adjusted, y, color=blue, s=23, label='DNA content + QC')
    ax.scatter(a.rho_adjusted_cycle, y, facecolors='white', edgecolors=orange,
               s=23, label='+ cycle proxy', zorder=3)
    ax.set(yticks=y, yticklabels=labels, xlim=(0, .48), ylim=(5.55, -.6),
           xlabel='Partial rank correlation', ylabel='Sorting plate (cells)')
    ax.set_title('Same-cell DDR score and γH2AX', loc='left', pad=12)
    ax.axvline(0, color=grey, lw=.6)
    ax.legend(loc='lower left', bbox_to_anchor=(-.08, -.49), fontsize=6.2,
              handletextpad=.4, labelspacing=.35)
    for i, row in enumerate(b.itertuples()):
        ay.plot([0, row.gammaH2AX_median_difference], [i, i], color=grey, lw=.8)
        ay.scatter(row.gammaH2AX_median_difference, i,
                   marker='o' if row.hras == 'control' else 's',
                   color=blue if row.hras == 'control' else orange, s=23)
    ay.tick_params(labelleft=False)
    ay.set(xlim=(0, 415), xlabel='Median γH2AX difference (a.u.)')
    ay.set_title('DDR-high versus other cells', loc='left', pad=12)
    ay.axvline(0, color=grey, lw=.6)
    ay.scatter([], [], color=blue, marker='o', s=20, label='HRAS uninduced')
    ay.scatter([], [], color=orange, marker='s', s=20, label='HRAS induced')
    ay.legend(loc='lower left', bbox_to_anchor=(-.08, -.49), fontsize=6.2,
              handletextpad=.4, labelspacing=.35)
    pathways = ['MMR', 'NER', 'DR_O6MeG', 'BER_8oxoG_C', 'BER_A_8oxoG',
                'BER_UG', 'BER_Hx', 'NHEJ']
    names = ['Mismatch repair', 'NER reporter reactivation', 'MGMT (O6-methylG)',
             'BER (8-oxoG:C)', 'BER (A:8-oxoG)', 'BER (U:G)', 'BER (hypoxanthine)',
             'End-joining reporter reactivation']
    for t, color, dy in [('2h', blue, -.13), ('24h', orange, .13)]:
        z = r[r.timepoint.eq(t)].set_index('endpoint').loc[pathways]
        yy = np.arange(8) + dy
        az.errorbar(z.rho, yy, xerr=np.vstack([z.rho-z.ci_low, z.ci_high-z.rho]),
                    fmt='o', color=color, ms=3.5, lw=.9, capsize=1.8,
                    label=f"{t[:-1]} h ({int(z.n_lines.min())} cell lines)")
        for j, (_, row) in enumerate(z.iterrows()):
            if row.q_signflip < .05:
                az.text(row.ci_high + .012, yy[j]+.04, '*', color=color,
                        fontsize=11, va='center')
    az.set(yticks=np.arange(8), yticklabels=names, ylim=(7.65, -.65),
           xlim=(-.18, .32), xlabel='Within-line partial rank correlation, equal-line Fisher-z mean')
    az.axvline(0, color=grey, lw=.65, ls='--')
    az.set_title('Same-cell DDR score and functional reporter readouts', loc='left', pad=12)
    az.legend(loc='upper right', fontsize=6.5)
    az.text(0, -.22, '* q < 0.05 across all 16 pathway × time comparisons; bars: 95% cell-line bootstrap CI',
            transform=az.transAxes, fontsize=6.5, va='top')
    for label, axis in [('a', ax), ('b', ay), ('c', az)]:
        axis.text(-.19 if label != 'c' else -.235, 1.12, label,
                  transform=axis.transAxes, fontsize=9, weight='bold', va='top')
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / 'Fig4_Direct_Validation.pdf')
    fig.savefig(args.output / 'Fig4_Direct_Validation.svg')
    fig.savefig(args.output / 'Fig4_Direct_Validation.png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
