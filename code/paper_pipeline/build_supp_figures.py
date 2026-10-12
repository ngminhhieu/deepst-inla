#!/usr/bin/env python3
"""Supplementary figures that can be rebuilt from ``source_data/`` alone.

Same figure system as ``build_natcom_figures.py`` (180 mm / 120 mm, Arial 5-7 pt,
validated model colours, fonttype 42):

* FigS_cross_scale_operating_points.pdf  <- source_data/FigS_cross_scale_operating_points.csv
* FigS_controlled_spatial_support.pdf    <- source_data/FigS_controlled_spatial_support.csv

FigS_commune_probability_comparison.pdf needs row-level (restricted) forecasts;
it is produced by ``export_commune_probabilities.py`` with the same palette.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import build_natcom_figures as B

SHORT = {  # source-data name -> (label, colour, marker)
    'INLA-TS': ('Structured Bayesian', B.GREEN, 's'),
    'TCN-GNN': ('Neural count', B.BLUE, '^'),
    'DeepST-Alert': ('Neural warning', B.ORANGE, 'D'),
    'DeepST-INLA': ('DeepST–INLA', B.PURPLE, 'o'),
}


def cross_scale():
    d = pd.read_csv(B.SRC / 'FigS_cross_scale_operating_points.csv')
    fig, axes = plt.subplots(1, 2, figsize=(B.W_MID, 52 * B.MM), sharex=True, sharey=True)
    # label offsets (points) chosen to avoid collisions
    off = {('Commune', 'INLA-TS'): (5, -4), ('Commune', 'TCN-GNN'): (5, 5),
           ('Commune', 'DeepST-INLA'): (5, 3), ('Commune', 'DeepST-Alert'): (-5, 5),
           ('Province', 'INLA-TS'): (5, -2), ('Province', 'TCN-GNN'): (5, 3),
           ('Province', 'DeepST-INLA'): (5, -3), ('Province', 'DeepST-Alert'): (5, -2)}
    for ax, scale, letter in zip(axes, ('Commune', 'Province'), 'ab'):
        part = d[d.scale == scale]
        for _, r in part.iterrows():
            lab, col, mk = SHORT[r.model]
            ax.scatter(r.false_alert_rate, r.outbreak_sensitivity, s=22, color=col,
                       marker=mk, zorder=3, edgecolor='white', linewidth=0.5)
            dx, dy = off[(scale, r.model)]
            ax.annotate(lab, (r.false_alert_rate, r.outbreak_sensitivity), xytext=(dx, dy),
                        textcoords='offset points', fontsize=6, color=B.INK,
                        ha='right' if dx < 0 else 'left', va='center')
        ax.set_xlim(-0.01, 0.30); ax.set_ylim(0, 1.0)
        ax.set_xlabel('False-alert rate (1 − specificity)')
        ax.grid(color=B.TRACK, lw=0.6, zorder=0)
        B.panel(ax, letter, scale, dx=-22 if letter == 'a' else -8, dy=6)
    axes[0].set_ylabel('Sensitivity')
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.2, top=0.88, wspace=0.08)
    B.save(fig, 'FigS_cross_scale_operating_points')


def controlled_support():
    d = pd.read_csv(B.SRC / 'FigS_controlled_spatial_support.csv')
    fig, axes = plt.subplots(1, 3, figsize=(B.W_FULL, 50 * B.MM))
    spec = [('brier_skill', 'Brier skill score', '↑ better'),
            ('auprc_lift_over_prevalence', 'AUPRC / prevalence', '↑ better'),
            ('crps_incidence', 'CRPS (incidence)', '↓ better')]
    for ax, (m, lab, note), letter in zip(axes, spec, 'abc'):
        ax.plot(d.median_communes_per_unit, d[m], color=B.PURPLE, lw=1.2, marker='o',
                ms=4, mec='white', mew=0.5, zorder=3)
        for _, r in d.iterrows():
            ax.annotate(f'{int(r.spatial_units)}', (r.median_communes_per_unit, r[m]),
                        xytext=(0, 5), textcoords='offset points', ha='center',
                        fontsize=5.5, color=B.MUTED)
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 4, 16, 102]); ax.set_xticklabels(['1', '4', '16', '102'])
        ax.set_xlim(0.7, 150); ax.minorticks_off()
        pad = 0.12 * (d[m].max() - d[m].min())
        ax.set_ylim(d[m].min() - pad, d[m].max() + 2.2 * pad)
        ax.set_xlabel('Median communes per spatial unit')
        ax.set_ylabel(lab)
        ax.grid(color=B.TRACK, lw=0.6, zorder=0)
        B.panel(ax, letter, note, dx=-24, dy=6)
    fig.text(0.995, 0.01, 'Numbers above points: spatial units', ha='right', va='bottom',
             fontsize=5.5, color=B.MUTED)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.86, wspace=0.34)
    B.save(fig, 'FigS_controlled_spatial_support')


def paired_forest():
    d = pd.read_csv(B.SRC / 'Uncertainty_paired_contrasts.csv')
    comps = [('TCN-GNN Base', 'Neural count'),
             ('Representation-INLA (variant)', 'Bayesian + neural repr.'), ('LSTM-GNN', 'LSTM\u2013GNN'),
             ('DeepST-Alert (proposed deep)', 'Neural warning'), ('INLA-TS', 'Structured Bayesian')]
    mets = [('outbreak_brier', 'Brier score', 'lower = DeepST\u2013INLA better'),
            ('crps', 'CRPS', 'lower = DeepST\u2013INLA better'),
            ('outbreak_f1', 'F1', 'higher = DeepST\u2013INLA better'),
            ('outbreak_sensitivity', 'Sensitivity', ''),
            ('outbreak_specificity', 'Specificity', ''),
            ('outbreak_precision', 'Precision', '')]
    fig, axs = plt.subplots(2, 3, figsize=(B.W_FULL, 80 * B.MM), sharey=True)
    y = np.arange(len(comps))[::-1]
    for i, (ax, (m, ttl, _)) in enumerate(zip(axs.flat, mets)):
        ax.axvline(0, color=B.FAINT, lw=0.8, zorder=1)
        for yi, (k, lab) in zip(y, comps):
            r = d[(d.model == k) & (d.metric == m)].iloc[0]
            sig = (r.ci025 > 0) or (r.ci975 < 0)
            ax.plot([r.ci025, r.ci975], [yi, yi], color=B.INK, lw=1.0, solid_capstyle='butt', zorder=2)
            ax.plot(r.difference_comparator_minus_hybrid, yi, 'o', ms=4, mec=B.INK, mew=0.9,
                    mfc=B.INK if sig else 'white', zorder=3)
        ax.set_yticks(y, [lab for _, lab in comps])
        ax.set_ylim(-0.6, len(comps) - 0.4)
        ax.set_xlabel('Comparator \u2212 DeepST\u2013INLA')
        B.quiet(ax, 'x')
        B.panel(ax, 'abcdef'[i], ttl, dx=-4 if i % 3 else -64)
        if i % 3:
            ax.tick_params(axis='y', length=0)
    fig.text(0.995, 0.005, 'Filled: 95% interval excludes zero. CRPS and Brier: positive = comparator worse.',
             ha='right', va='bottom', fontsize=5.8, color=B.MUTED)
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.12, top=0.92, wspace=0.12, hspace=0.62)
    B.save(fig, 'FigS_paired_contrasts')


if __name__ == '__main__':
    paired_forest()
    cross_scale()
    controlled_support()
    print('supplementary figures written to', B.FIG)
