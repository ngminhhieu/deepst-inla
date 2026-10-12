#!/usr/bin/env python3
"""Main-text figures (Figs 1, 3-8) in a Nature Communications figure system.

Design rules
------------
* Final artwork size: 180 mm (double column) wide unless stated; Arial/Helvetica
  (Liberation Sans as metric-compatible fallback), 5-7 pt text, 8 pt bold panel
  letters, vector PDF with embedded TrueType (fonttype 42).
* No in-figure titles (the caption carries the message); direct labels instead of
  legends wherever possible; thin, quiet axes; no gridlines except where they
  help reading values.
* One colour per modelling system, shared with Fig. 2 and validated for
  colour-vision deficiency (see MODEL_COLORS). Marker shapes repeat the identity
  so colour is never the only cue.

All numbers are read from ``source_data/`` (no restricted data needed).
Figure 2 (framework schematic) is produced by ``build_fig2_framework.py``.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.ticker import FuncFormatter, MultipleLocator

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'source_data'
FIG = ROOT / 'paper' / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
W_FULL = 180 * MM          # Nature Communications double-column width
W_MID = 120 * MM

# ------------------------------------------------------------------ style
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 6.5,
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.major.pad': 2,
    'ytick.major.pad': 2,
    'axes.labelpad': 2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.solid_capstyle': 'round',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 600,
})

INK = '#1F2430'
MUTED = '#5C6470'
FAINT = '#8A919C'
RULE = '#C9CED6'
TRACK = '#ECEEF1'

# Validated with the categorical-palette checker (all pairs, light surface).
PURPLE, BLUE, GREEN, ORANGE = '#7B2D8E', '#3F74AD', '#009E73', '#D55E00'

SYSTEMS = {  # source-data model name -> (display name, colour, marker)
    'INLA-TS': ('Structured Bayesian', GREEN, 's'),
    'TCN-GNN Base': ('Neural count', BLUE, '^'),
    'Representation-INLA (variant)': ('Bayesian + neural representation', ORANGE, 'D'),
    'DeepST-INLA (proposed hybrid)': ('DeepST–INLA', PURPLE, 'o'),
}
ORDER = list(SYSTEMS)  # draw order: hybrid last (on top)
HYB = 'DeepST-INLA (proposed hybrid)'


def unc():
    """Bootstrap s.d. (target-month blocks) keyed by (source-data model, scope, metric)."""
    d = pd.read_csv(SRC / 'Uncertainty_model_metric_intervals.csv')
    return d.set_index(['model', 'scope', 'metric']).bootstrap_sd


def save(fig, stem):
    fig.savefig(FIG / f'{stem}.pdf')
    plt.close(fig)


def panel(ax, letter, title='', dx=-22, dy=6):
    """Bold 8 pt panel letter followed by a 6.5 pt title, on one baseline."""
    ax.annotate(letter, xy=(0, 1), xycoords='axes fraction', xytext=(dx, dy),
                textcoords='offset points', fontsize=8, fontweight='bold',
                ha='left', va='bottom', annotation_clip=False)
    if title:
        ax.annotate(title, xy=(0, 1), xycoords='axes fraction', xytext=(dx + 9, dy + 0.6),
                    textcoords='offset points', fontsize=6.8, color=INK,
                    ha='left', va='bottom', annotation_clip=False)


def quiet(ax, grid=None):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors=INK)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK)
    if grid:
        ax.grid(axis=grid, color=TRACK, linewidth=0.6)
        ax.set_axisbelow(True)


def better_arrow(ax, start, end, text, text_xy, ha='center'):
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle='-|>',
                                 mutation_scale=6, lw=0.8, color=FAINT, shrinkA=0, shrinkB=0))
    ax.text(*text_xy, text, transform=ax.transAxes, fontsize=5.8, color=FAINT,
            ha=ha, va='center', style='italic')


# ---------------------------------------------------------------- Fig. 1
def fig1():
    cent = pd.read_csv(SRC / 'Fig1_commune_centroids.csv')
    burden = pd.read_csv(SRC / 'Fig1_province_burden.csv')
    monthly = pd.read_csv(SRC / 'Fig1_monthly_cases.csv', parse_dates=['date'])
    prov = ['Tay Ninh', 'Dong Thap', 'Vinh Long', 'An Giang', 'Can Tho', 'Ca Mau']
    pcol = dict(zip(prov, ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00']))
    n_com = cent.groupby('province').size()
    forecasts = 99720

    fig = plt.figure(figsize=(W_FULL, 62 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.82, 1.55], left=0.045, right=0.99,
                          bottom=0.17, top=0.82, wspace=0.50)
    ax_m, ax_b, ax_t = (fig.add_subplot(gs[0, i]) for i in range(3))

    # a: map
    for p in prov:
        d = cent[cent.province == p]
        ax_m.scatter(d.longitude, d.latitude, s=5.5, color=pcol[p], linewidths=0, zorder=2)
    ax_m.set_aspect('equal', adjustable='datalim')
    ax_m.set_xlim(104.35, 106.9); ax_m.set_ylim(8.35, 11.15)
    ax_m.set_xticks([105, 106]); ax_m.set_yticks([9, 10, 11])
    ax_m.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}°E'))
    ax_m.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}°N'))
    quiet(ax_m)
    panel(ax_m, 'a', f'Study communes (n = {len(cent)})', dx=-20)

    # b: burden (bars share the map colours and act as its legend)
    b = burden.set_index('province').loc[prov]
    y = np.arange(len(prov))
    ax_b.barh(y, b.cases_k, color=[pcol[p] for p in prov], height=0.62, linewidth=0)
    ax_b.set_yticks(y, [f'{p} ({n_com[p]})' for p in prov])
    ax_b.invert_yaxis()
    for yi, v in zip(y, b.cases_k):
        ax_b.text(v + 2, yi, f'{v:.0f}k', va='center', fontsize=6, color=INK)
    ax_b.set_xlim(0, 140); ax_b.set_xticks([0, 50, 100])
    ax_b.set_xlabel('Reported cases, 2010–2025 (thousands)')
    ax_b.tick_params(axis='y', length=0)
    quiet(ax_b)
    panel(ax_b, 'b', 'Observed burden (communes)', dx=-58)

    # c: temporal split
    t0, t1, t2, t3 = (pd.Timestamp(s) for s in ('2010-01-01', '2019-01-01', '2021-01-01', '2026-01-01'))
    bands = [(t0, t1, 'Train', '#E9EDF2', '#C5CDD8'), (t1, t2, 'Val.', '#D6DCE5', '#8E9BB0'),
             (t2, t3, 'Locked test', '#F3E5F5', PURPLE)]
    ymax = 15.5
    for x0, x1, lab, fill, strip in bands:
        ax_t.axvspan(x0, x1, color=fill, lw=0, zorder=0)
        ax_t.add_patch(Rectangle((mpl.dates.date2num(x0), 1.02), mpl.dates.date2num(x1) - mpl.dates.date2num(x0), 0.11,
                                 transform=ax_t.get_xaxis_transform(), clip_on=False, color=strip, lw=0))
        ax_t.text(x0 + (x1 - x0) / 2, 1.075, lab, transform=ax_t.get_xaxis_transform(), ha='center', va='center',
                  fontsize=6, color='white' if lab != 'Train' else INK, fontweight='bold')
    ax_t.plot(monthly.date, monthly.cases / 1000, color=INK, lw=1.1, zorder=3)
    ax_t.set_xlim(t0, pd.Timestamp('2025-12-31')); ax_t.set_ylim(0, ymax)
    ax_t.set_xticks([pd.Timestamp(f'{y}-01-01') for y in range(2010, 2026, 3)])
    ax_t.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y'))
    ax_t.set_yticks([0, 5, 10, 15])
    ax_t.set_ylabel('Monthly cases, all communes (thousands)')
    ax_t.set_xlabel('Year')
    quiet(ax_t)
    panel(ax_t, 'c', f'Locked temporal split ({forecasts:,} test forecasts)', dx=-24, dy=17)
    save(fig, 'Fig1_study_design')


# ---------------------------------------------------------------- Fig. 3
def fig3():
    d = pd.read_csv(SRC / 'Fig3_spatial_support.csv')
    xs = np.arange(len(d))
    ticks = [f'{int(u)}\n({int(m)})' for u, m in zip(d.spatial_units, d.median_communes_per_unit)]
    fig, axs = plt.subplots(1, 2, figsize=(W_MID, 50 * MM),
                            gridspec_kw=dict(left=0.115, right=0.985, bottom=0.27, top=0.83, wspace=0.42))
    for ax, col, lab, (lo, hi), ttl in [
        (axs[0], 'crps_incidence', 'CRPS per 100,000 ↓', (3.4, 8.6), 'Distributional error'),
        (axs[1], 'mae_incidence', 'MAE per 100,000 ↓', (2.6, 12.4), 'Mean absolute error'),
    ]:
        v = d[col].to_numpy()
        bs = pd.read_csv(SRC / 'Uncertainty_spatial_support_bootstrap.csv').set_index(['scale_a', 'metric'])
        names = ['support_128', 'support_32', 'province_6']
        for i, nm in enumerate(names, start=1):  # paired-difference interval anchored at the commune value
            r = bs.loc[(nm, col)]
            ax.plot([xs[i] - 0.0] * 2, [v[0] + r.ci_lower, v[0] + r.ci_upper], color=BLUE, lw=0.9, alpha=0.55,
                    solid_capstyle='butt', zorder=1)
            for yy in (v[0] + r.ci_lower, v[0] + r.ci_upper):
                ax.plot([xs[i] - 0.06, xs[i] + 0.06], [yy, yy], color=BLUE, lw=0.9, alpha=0.55, zorder=1)
        ax.plot(xs, v, color=BLUE, lw=1.4, zorder=2)
        ax.scatter(xs, v, s=16, color=BLUE, edgecolor='white', linewidth=0.6, zorder=3)
        for i, val in enumerate(v):
            ax.text(xs[i] + 0.2, val + (hi - lo) * 0.02, f'{val:.2f}', ha='left', va='bottom',
                    fontsize=6, color=INK)
            if i:
                ylo = v[0] + bs.loc[(names[i - 1], col)].ci_lower
                ax.text(xs[i], ylo - (hi - lo) * 0.025, f'{100 * (val / v[0] - 1):.0f}%'.replace('-', '−'),
                        ha='center', va='top', fontsize=5.8, color=MUTED)
        ax.set_xticks(xs, ticks)
        ax.set_xlim(-0.35, len(d) - 0.1); ax.set_ylim(lo, hi)
        ax.set_ylabel(lab)
        ax.set_xlabel('Prediction units\n(median communes per unit)', labelpad=3)
        quiet(ax, 'y')
    panel(axs[0], 'a', 'Distributional error', dx=-25)
    panel(axs[1], 'b', 'Mean absolute error', dx=-25)
    save(fig, 'Fig3_spatial_support')


# ---------------------------------------------------------------- Fig. 4
def fig4():
    m = pd.read_csv(SRC / 'Fig4_tradeoff.csv').set_index('model')
    m['far'] = 1 - m.outbreak_specificity
    U = unc()
    sdcol = {'outbreak_brier': 'outbreak_brier', 'far': 'outbreak_specificity',
             'outbreak_precision': 'outbreak_precision'}
    # Leave enough horizontal room for the long Representation--INLA labels.
    specs = [('outbreak_brier', 'Brier score ↓', (0.060, 0.188)),
             ('far', 'False-alert rate ↓', (-0.025, 0.285)),
             ('outbreak_precision', 'Precision ↑', (0.28, 0.88))]
    ttl = ['Event probability vs detection', 'False alerts vs detection', 'Alert operating point']
    fig, axs = plt.subplots(1, 3, figsize=(W_FULL, 55 * MM), sharey=True,
                            gridspec_kw=dict(left=0.058, right=0.99, bottom=0.2, top=0.84, wspace=0.12))
    offs = {  # label offsets (points) per panel and system
        0: {'INLA-TS': (4, 3), 'TCN-GNN Base': (4, 3), HYB: (6, -2), 'Representation-INLA (variant)': (-6, -9)},
        1: {'INLA-TS': (5, 3), 'TCN-GNN Base': (5, 3), HYB: (6, -2), 'Representation-INLA (variant)': (-6, -9)},
        2: {'INLA-TS': (-5, 5), 'TCN-GNN Base': (-5, 6), HYB: (-5, 6), 'Representation-INLA (variant)': (6, -2)},
    }
    ha = {0: {'Representation-INLA (variant)': 'right'}, 1: {'Representation-INLA (variant)': 'right'},
          2: {'INLA-TS': 'right', 'TCN-GNN Base': 'right', HYB: 'right'}}
    for i, (ax, (col, lab, xl)) in enumerate(zip(axs, specs)):
        for k in ORDER:
            nm, c, mk = SYSTEMS[k]
            x, y = m.loc[k, col], m.loc[k, 'outbreak_sensitivity']
            ax.errorbar(x, y, xerr=U[(k, 'mean_H1_H3', sdcol[col])],
                        yerr=U[(k, 'mean_H1_H3', 'outbreak_sensitivity')], fmt='none',
                        ecolor=c, elinewidth=0.7, capsize=0, alpha=0.55, zorder=2)
            ax.scatter(x, y, s=36 if k == HYB else {'^': 30, 'D': 24, 's': 24}[mk], marker=mk, color=c,
                       edgecolor='white', linewidth=0.7, zorder=3)
            dx, dy = offs[i][k]
            ax.annotate(nm, (x, y), xytext=(dx, dy), textcoords='offset points', fontsize=6,
                        color=c if k == HYB else INK, fontweight='bold' if k == HYB else 'normal',
                        ha=ha[i].get(k, 'left'), va='center')
        ax.set_xlim(*xl); ax.set_ylim(0.12, 0.95)
        ax.set_xlabel(lab)
        quiet(ax, 'both')
        panel(ax, 'abc'[i], ttl[i], dx=-26 if i == 0 else -8)
    axs[0].set_ylabel('Sensitivity ↑')
    axs[1].tick_params(axis='y', length=0); axs[2].tick_params(axis='y', length=0)
    # 'better' cues sit in the empty corner of each panel
    for ax, tail, head in [(axs[0], (0.93, 0.17), (0.82, 0.29)), (axs[1], (0.93, 0.17), (0.82, 0.29)),
                           (axs[2], (0.10, 0.17), (0.21, 0.29))]:
        better_arrow(ax, tail, head, 'better', (tail[0], 0.095))
    save(fig, 'Fig4_tradeoff')


# ---------------------------------------------------------------- Fig. 5
def fig5():
    b = pd.read_csv(SRC / 'Fig5_horizon_performance.csv')
    mets = [('crps', 'CRPS ↓', 'Incidence'), ('outbreak_brier', 'Brier score ↓', 'Event probability'),
            ('outbreak_sensitivity', 'Sensitivity ↑', 'Outbreak detection'), ('outbreak_f1', 'F1 ↑', 'Alert balance')]
    U = unc()
    fig, axs = plt.subplots(1, 4, figsize=(W_FULL, 48 * MM),
                            gridspec_kw=dict(left=0.058, right=0.99, bottom=0.28, top=0.82, wspace=0.55))
    for i, (ax, (col, ylab, ttl)) in enumerate(zip(axs, mets)):
        for k in ORDER:
            nm, c, mk = SYSTEMS[k]
            g = b[b.model == k].sort_values('horizon')
            hyb = k == HYB
            dodge = {'INLA-TS': -0.09, 'TCN-GNN Base': -0.03, 'Representation-INLA (variant)': 0.03, HYB: 0.09}[k]
            ax.errorbar(g.horizon + dodge, g[col], yerr=[U[(k, f'H{int(h)}', col)] for h in g.horizon],
                        fmt='none', ecolor=c, elinewidth=0.6, capsize=0, alpha=0.5, zorder=1)
            ax.plot(g.horizon + dodge, g[col], color=c, lw=1.7 if hyb else 1.1, zorder=3 if hyb else 2,
                    marker=mk, ms=4.2 if hyb else 3.4, mec='white', mew=0.5)
        ax.set_xticks([1, 2, 3], ['H1', 'H2', 'H3'])
        ax.set_xlim(0.75, 3.25)
        ax.set_ylabel(ylab)
        quiet(ax, 'y')
        panel(ax, 'abcd'[i], ttl, dx=-27)
    axs[0].set_yticks([4, 6, 8, 10]); axs[0].set_ylim(3.8, 10.8)
    axs[1].set_yticks([0.05, 0.10, 0.15, 0.20]); axs[1].set_ylim(0.04, 0.215)
    axs[2].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0]); axs[2].set_ylim(0.1, 1.0)
    axs[3].set_yticks([0.3, 0.4, 0.5, 0.6, 0.7]); axs[3].set_ylim(0.25, 0.75)
    handles = [Line2D([0], [0], color=SYSTEMS[k][1], lw=1.4, marker=SYSTEMS[k][2], ms=4, mec='white', mew=0.4,
                      label=SYSTEMS[k][0]) for k in ORDER]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02),
               handlelength=2.0, columnspacing=2.2, handletextpad=0.5)
    fig.text(0.5, 0.135, 'Forecast horizon (months ahead)', ha='center', fontsize=7, color=INK)
    save(fig, 'Fig5_horizon_performance')


# ---------------------------------------------------------------- Fig. 6
def fig6():
    d = pd.read_csv(SRC / 'Fig6_operating_point.csv').set_index('model')
    keys = ['INLA-TS', HYB, 'Representation-INLA (variant)']
    U = unc()
    sdm = {'detected_per_100_outbreaks': 'outbreak_sensitivity', 'false_alerts_per_100_nonoutbreaks': 'outbreak_specificity'}
    fig, axs = plt.subplots(1, 2, figsize=(W_FULL, 38 * MM), sharey=True,
                            gridspec_kw=dict(left=0.185, right=0.985, bottom=0.25, top=0.80, wspace=0.12))
    y = np.arange(len(keys))
    specs = [(axs[0], 'detected_per_100_outbreaks', 'outbreak_tp', (0, 118), [0, 25, 50, 75, 100],
              'Observed outbreaks detected per 100 ↑', 'Event recovery'),
             (axs[1], 'false_alerts_per_100_nonoutbreaks', 'outbreak_fp', (0, 36), [0, 5, 10, 15, 20, 25, 30],
              'False alerts per 100 non-outbreaks ↓', 'Operational burden')]
    for ax, col, cnt, xl, tk, xlab, ttl in specs:
        for yi, k in zip(y, keys):
            nm, c, _ = SYSTEMS[k]
            v = d.loc[k, col]
            ax.barh(yi, v, height=0.52, color=c, linewidth=0, zorder=2)
            e = 100 * U[(k, 'mean_H1_H3', sdm[col])]
            ax.errorbar(v, yi, xerr=e, fmt='none', ecolor=INK, elinewidth=0.7, capsize=1.5, capthick=0.7, zorder=3)
            ax.annotate(f'{v:.1f}', (v + e, yi), xytext=(3, 0), textcoords='offset points', va='center',
                        ha='left', fontsize=6.4, fontweight='bold', color=INK)
            ax.annotate(f'({int(d.loc[k, cnt]):,})', (v + e, yi), xytext=(20, 0), textcoords='offset points',
                        va='center', ha='left', fontsize=5.8, color=MUTED)
        ax.set_xlim(*xl); ax.set_xticks(tk)
        ax.set_xlabel(xlab)
        quiet(ax)
        ax.tick_params(axis='y', left=False)
    axs[0].set_yticks(y, [SYSTEMS[k][0] for k in keys])
    axs[0].invert_yaxis()
    axs[0].tick_params(axis='y', length=0)
    for lab, k in zip(axs[0].get_yticklabels(), keys):
        if k == HYB:
            lab.set_fontweight('bold')
    panel(axs[0], 'a', 'Event recovery', dx=-84)
    panel(axs[1], 'b', 'Operational burden', dx=-8)
    save(fig, 'Fig6_operating_point')


# ---------------------------------------------------------------- Fig. 7
def fig7():
    r = pd.read_csv(SRC / 'Fig7_temporal_robustness.csv')
    labs = {'MAE': 'MAE ↓', 'RMSE': 'RMSE ↓', 'CRPS': 'CRPS ↓', 'Brier': 'Brier ↓', 'F1': 'F1 ↑'}
    y = np.arange(len(r))
    fig, axs = plt.subplots(1, 2, figsize=(W_FULL, 42 * MM), sharey=True,
                            gridspec_kw=dict(left=0.075, right=0.985, bottom=0.25, top=0.80, wspace=0.20))
    ax = axs[0]
    ax.axvline(1, color=FAINT, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for yi, v in zip(y, r.later_to_full_ratio):
        ax.hlines(yi, v, 1, color=RULE, lw=1.6, zorder=1)
        ax.scatter(1, yi, s=14, facecolor='white', edgecolor=FAINT, linewidth=0.8, zorder=2)
        ax.scatter(v, yi, s=26, color=PURPLE, edgecolor='white', linewidth=0.6, zorder=3)
        ax.text(v - 0.012, yi, f'{v:.2f}', ha='right', va='center', fontsize=6.2, color=INK)
    ax.set_xlim(0.62, 1.04); ax.set_xticks([0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel('2023–2025 relative to 2021–2025 (ratio)')
    ax = axs[1]
    for yi, v in zip(y, r.rolling_refit_deterioration_percent):
        ax.hlines(yi, 0, v, color=RULE, lw=1.6, zorder=1)
        ax.scatter(v, yi, s=26, color=PURPLE, edgecolor='white', linewidth=0.6, zorder=3)
        ax.text(v + 0.12, yi, f'{v:.1f}%', ha='left', va='center', fontsize=6.2, color=INK)
    ax.axvline(0, color=INK, lw=0.8, zorder=1)
    ax.set_xlim(0, 5.2); ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xlabel('Deterioration under causal rolling refit (%)')
    axs[0].set_yticks(y, [labs[m] for m in r.metric]); axs[0].invert_yaxis()
    axs[1].set_yticks(y, [labs[m] for m in r.metric])
    for a in axs:
        quiet(a)
        a.tick_params(axis='y', length=0)
    panel(axs[0], 'a', 'Later-period sensitivity analysis', dx=-30)
    panel(axs[1], 'b', 'Rolling refit, 62 forecast vintages', dx=-30)
    save(fig, 'Fig7_robustness')


# ---------------------------------------------------------------- Fig. 8
def fig8():
    ar = pd.read_csv(SRC / 'Fig8_architecture_ablation.csv')
    fe = pd.read_csv(SRC / 'Fig8_feature_ablation.csv')
    pretty = {'TCN-GNN': 'TCN–GNN (selected)', 'TCN-GNN + dynamic spatial gate': 'TCN–GNN +\ndynamic spatial gate',
              'CNN-LSTM-GNN': 'CNN–LSTM–GNN', 'LSTM-GNN': 'LSTM–GNN', 'Last-state GNN': 'Last-state GNN',
              'Full set': 'Full set (selected)', 'Basic + soil moisture': 'Basic + soil moisture', 'Basic + ET0': 'Basic + ET0',
              'Basic + solar radiation': 'Basic + solar radiation', 'Basic + dew point': 'Basic + dew point',
              'Basic climate': 'Basic climate'}
    fig = plt.figure(figsize=(W_FULL, 56 * MM))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.50, 1, 1], left=0.176, right=0.985, bottom=0.22, top=0.80,
                          wspace=0.22)
    # axes: (a-crps, a-f1, spacer, b-crps, b-f1); the spacer is empty and holds the panel-b labels
    groups = [(ar, 'architecture', (gs[0, 0], gs[0, 1]), 'a', 'Temporal–graph architecture'),
              (fe, 'feature_set', (gs[0, 3], gs[0, 4]), 'b', 'Environmental feature set')]
    first_axes = []
    for frame, labcol, (g0, g1), letter, ttl in groups:
        n = len(frame)
        y = np.arange(n)
        sel = frame[labcol].isin(['TCN-GNN', 'Full set']).to_numpy()
        axg = []
        for j, (g, rawcol, xl, tk, hdr) in enumerate([
            (g0, 'crps', (6.68, 6.99), [6.7, 6.8, 6.9], 'CRPS ↓'),
            (g1, 'f1', (0.51, 0.59), [0.52, 0.55, 0.58], 'F1 ↑')]):
            ax = fig.add_subplot(g, sharey=axg[0] if axg else None)
            mu = frame[rawcol].to_numpy()
            sd = frame[rawcol + '_sd'].to_numpy()
            for yi, s_ in zip(y, sel):
                if s_:
                    ax.axhspan(yi - 0.5, yi + 0.5, color='#F2E6F5', lw=0, zorder=0)
            ref = mu[sel][0]
            ax.axvline(ref, color=PURPLE, lw=0.6, ls=(0, (2, 2)), alpha=0.6, zorder=1)
            for yi, m_, s2, s_ in zip(y, mu, sd, sel):
                c_ = PURPLE if s_ else '#6B7280'
                ax.plot([m_ - s2, m_ + s2], [yi, yi], color=c_, lw=1.0, solid_capstyle='butt', zorder=2)
                ax.plot(m_, yi, 'o', ms=4.2 if s_ else 3.6, color=c_, mec='white', mew=0.6, zorder=3)
            ax.set_xlim(*xl); ax.set_xticks(tk)
            ax.set_ylim(n - 0.5, -0.5)
            ax.set_title(hdr, fontsize=6.4, color=INK, pad=3)
            quiet(ax, 'x')
            ax.spines['left'].set_visible(j == 0)
            ax.tick_params(axis='y', length=0, pad=7)
            if j == 0:
                ax.set_yticks(y, [pretty.get(x, x) for x in frame[labcol]])
                for tl, s in zip(ax.get_yticklabels(), sel):
                    if s:
                        tl.set_fontweight('bold')
            else:
                ax.tick_params(labelleft=False)
            axg.append(ax)
        first_axes.append(axg[0])
        axg[0].figure.text(0, 0, '')  # keep tight bbox stable
        panel(axg[0], letter, ttl, dx=-(88 if letter == 'a' else 79), dy=14)
    fig.text(0.58, 0.04, 'Validation score, mean ± s.d. over five training seeds', ha='center', fontsize=7, color=INK)
    save(fig, 'Fig8_ablation')


if __name__ == '__main__':
    for f in (fig1, fig3, fig4, fig5, fig6, fig7, fig8):
        f()
    print('Figures written to', FIG)
