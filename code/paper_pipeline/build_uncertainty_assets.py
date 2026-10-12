#!/usr/bin/env python3
"""Uncertainty tables and source data (target-month block bootstrap + five-seed SD).

Policy: the locked point estimates in ``source_data/Table1_model_summary.csv`` and
``Fig5_horizon_performance.csv`` stay the headline numbers. The complete
uncertainty rerun (``results/uncertainty_complete``) supplies the *uncertainty*:
the s.d. of 2,000 paired target-month block-bootstrap replicates (all communes and
horizons of a month stay together) and the paired contrasts against DeepST-INLA.
Seed s.d. comes from ``results/uncertainty_all_experiments`` (5 seeds, 17 neural
configurations; test period).

Writes: source_data/Uncertainty_*.csv, paper/generated/table_uncertainty_*.tex
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC, GEN = ROOT / 'source_data', ROOT / 'paper' / 'generated'
UC = ROOT / 'results' / 'uncertainty_complete'
UP = ROOT / 'results' / 'uncertainty_paper_20261004'
SEED = ROOT / 'results' / 'uncertainty_all_experiments'
AUDITED_SEED = ROOT / 'results' / 'rev_20261012' / 'seed_suite_audited'
GEN.mkdir(parents=True, exist_ok=True)

NAME = {  # bootstrap model name -> manuscript source-data name
    'INLA-TS': 'INLA-TS', 'TCN-GNN-Base': 'TCN-GNN Base', 'LSTM-GNN': 'LSTM-GNN',
    'DeepST-Alert': 'DeepST-Alert (proposed deep)',
    'TCN-GNN-Deep-Ensemble': 'TCN-GNN Deep Ensemble (variant)',
    'Representation-INLA': 'Representation-INLA (variant)',
    'DeepST-INLA': 'DeepST-INLA (proposed hybrid)'}
LABEL = {  # table row labels (as in Table 1)
    'INLA-TS': 'Structured Bayesian model', 'TCN-GNN Base': 'Neural count model',
    'LSTM-GNN': 'LSTM--GNN (variant)', 'DeepST-Alert (proposed deep)': 'Neural warning model',
    'TCN-GNN Deep Ensemble (variant)': 'Neural ensemble',
    'Representation-INLA (variant)': 'Bayesian + neural representation',
    'DeepST-INLA (proposed hybrid)': 'DeepST--INLA'}
T1_ORDER = ['DeepST-INLA (proposed hybrid)', 'Representation-INLA (variant)',
            'TCN-GNN Base', 'DeepST-Alert (proposed deep)', 'INLA-TS']
ALL_ORDER = T1_ORDER + ['LSTM-GNN']
MET = [('crps', 'CRPS', 3), ('outbreak_brier', 'Brier', 3), ('outbreak_sensitivity', 'Sensitivity', 3),
       ('outbreak_specificity', 'Specificity', 3), ('outbreak_f1', 'F1', 3)]
MINUS = '$-$'


def fnum(x, d=3):
    s = f'{x:.{d}f}'
    return s.replace('-', MINUS)


def pm(est, sd, d=3):
    return f'{fnum(est, d)}\\,$\\pm$\\,{sd:.{d}f}'


def texify(n):
    return (n.replace('CNN-LSTM-GNN', 'CNN--LSTM--GNN').replace('TCN-GNN', 'TCN--GNN')
            .replace('LSTM-GNN', 'LSTM--GNN'))


def write_tex(name, body):
    (GEN / name).write_text(body)


def locked():
    t1 = pd.read_csv(SRC / 'Table1_model_summary.csv').set_index('model')
    h = pd.read_csv(SRC / 'Fig5_horizon_performance.csv')
    out = {}
    for m in t1.index:
        out[(m, 'mean_H1_H3')] = t1.loc[m]
    for _, r in h.iterrows():
        out[(r.model, f'H{int(r.horizon)}')] = r
    return out


def main():
    iv = pd.read_csv(UC / 'bootstrap' / 'model_metric_intervals.csv')
    iv['model'] = iv.model.map(NAME)
    iv = iv[iv.model.ne('TCN-GNN Deep Ensemble (variant)')].copy()
    lk = locked()
    iv['locked_estimate'] = [lk[(m, s)][c] if (m, s) in lk else np.nan
                             for m, s, c in zip(iv.model, iv.scope, iv.metric)]
    iv = iv.rename(columns={'estimate': 'rerun_estimate', 'bootstrap_sd': 'bootstrap_sd',
                            'ci025': 'rerun_ci025', 'ci975': 'rerun_ci975'})
    iv['display_estimate'] = iv.locked_estimate.fillna(iv.rerun_estimate)
    iv.to_csv(SRC / 'Uncertainty_model_metric_intervals.csv', index=False)
    sd = iv.set_index(['model', 'scope', 'metric'])

    # ---- Table 1 body (locked estimate +/- bootstrap s.d.); bold = best locked value per column
    best = {}
    for c, _, _ in MET:
        vals = {m: lk[(m, 'mean_H1_H3')][c] for m in T1_ORDER}
        best[c] = (min if c in ('crps', 'outbreak_brier') else max)(vals, key=vals.get)
    rows = []
    for m in T1_ORDER:
        cells = []
        for c, _, d in MET:
            v, s = lk[(m, 'mean_H1_H3')][c], sd.loc[(m, 'mean_H1_H3', c), 'bootstrap_sd']
            txt = f'{v:.{d}f}'
            if best[c] == m:
                txt = f'\\textbf{{{txt}}}'
            cells.append(f'{txt}\\,$\\pm$\\,{s:.{d}f}')
        rows.append(f'{LABEL[m]} & ' + ' & '.join(cells) + '\\\\')
    write_tex('table_main_uncertainty_tabular.tex',
              '\\resizebox{\\textwidth}{!}{\\begin{tabular}{lccccc}\n\\toprule\nModel & CRPS & Brier & Sensitivity & Specificity & F1\\\\\n\\midrule\n'
              + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}}\n')

    # ---- Supplementary: by-scope table (rerun estimate with SD would mislead -> locked +/- SD for 6, rerun for LSTM-GNN)
    lines = ['\\begin{table}[H]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}',
             '\\caption{Target-month block-bootstrap uncertainty for every system and horizon (locked 2021--2025 test period). '
             'Entries are the locked point estimate $\\pm$ the s.d. of 2,000 paired bootstrap replicates of the 60 target months '
             '(all communes and horizons in a month stay together). LSTM--GNN was not part of the locked comparison, so its row shows the '
             'complete-rerun estimate. Percentile 95\\% intervals are in the source data.}'
             '\\label{tab:s_uncertainty_scope}',
             '\\begin{tabular}{llccccc}', '\\toprule',
             'Scope & Model & CRPS & Brier & Sensitivity & Specificity & F1 \\\\', '\\midrule']
    for s, sl in [('H1', 'H1'), ('H2', 'H2'), ('H3', 'H3'), ('mean_H1_H3', 'Mean H1--H3')]:
        for i, m in enumerate(ALL_ORDER):
            cells = []
            for c, _, d in MET:
                r = sd.loc[(m, s, c)]
                est = lk[(m, s)][c] if (m, s) in lk else r.rerun_estimate
                cells.append(pm(est, r.bootstrap_sd, d))
            lines.append(f'{sl if i == 0 else ""} & {LABEL[m]} & ' + ' & '.join(cells) + '\\\\')
        if s != 'mean_H1_H3':
            lines.append('\\midrule')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    write_tex('table_uncertainty_scope.tex', '\n'.join(lines) + '\n')

    # ---- paired contrasts vs DeepST-INLA (rerun)
    pc = pd.read_csv(UC / 'bootstrap' / 'paired_contrasts_vs_deepst_inla.csv')
    pc['model'] = pc.model.map(NAME)
    pc = pc[pc.model.ne('TCN-GNN Deep Ensemble (variant)')].copy()
    point = iv.set_index(['model', 'scope', 'metric'])
    pc['difference_comparator_minus_hybrid'] = [
        point.loc[(m, 'mean_H1_H3', metric), 'display_estimate'] -
        point.loc[('DeepST-INLA (proposed hybrid)', 'mean_H1_H3', metric), 'display_estimate']
        for m, metric in zip(pc.model, pc.metric)
    ]
    pc.drop(columns=['estimate']).to_csv(SRC / 'Uncertainty_paired_contrasts.csv', index=False)
    pcm = pc.set_index(['model', 'metric'])
    cols = [('outbreak_brier', 'Brier'), ('crps', 'CRPS'), ('outbreak_sensitivity', 'Sensitivity'),
            ('outbreak_specificity', 'Specificity'), ('outbreak_f1', 'F1')]
    L = ['\\begin{table}[H]\\centering\\scriptsize\\setlength{\\tabcolsep}{2.6pt}',
         '\\caption{Paired differences from DeepST--INLA (comparator minus DeepST--INLA) on the locked 2021--2025 test period, mean H1--H3. '
         'Values are the locked point difference with the percentile 95\\% interval from 2,000 paired target-month block-bootstrap replicates. '
         'Positive CRPS and Brier differences mean the comparator was worse; positive sensitivity, specificity and F1 differences mean it was higher. '
         'Intervals that exclude zero are in bold.}\\label{tab:s_paired}',
         '\\resizebox{\\textwidth}{!}{\\begin{tabular}{l' + 'c' * len(cols) + '}', '\\toprule',
         'Comparator & ' + ' & '.join(n for _, n in cols) + '\\\\', '\\midrule']
    for m in ['TCN-GNN Base', 'Representation-INLA (variant)',
              'LSTM-GNN', 'DeepST-Alert (proposed deep)', 'INLA-TS']:
        cells = []
        for c, _ in cols:
            r = pcm.loc[(m, c)]
            # Report the locked point difference; bootstrap replicates supply only the interval.
            est_value = (point.loc[(m, 'mean_H1_H3', c), 'display_estimate'] -
                         point.loc[('DeepST-INLA (proposed hybrid)', 'mean_H1_H3', c), 'display_estimate'])
            est, ci = fnum(est_value), f'({fnum(r.ci025)}, {fnum(r.ci975)})'
            if r.ci025 > 0 or r.ci975 < 0:
                est, ci = f'\\textbf{{{est}}}', f'\\textbf{{{ci}}}'
            t = f'\\shortstack{{{est}\\\\{ci}}}'
            cells.append(t)
        paired_label = {'TCN-GNN Base': 'Neural count model'}.get(m, LABEL[m])
        L.append(f'{paired_label} & ' + ' & '.join(cells) + '\\\\')
    L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
    write_tex('table_uncertainty_paired.tex', '\n'.join(L) + '\n')

    # ---- five-seed table for all 17 neural configurations
    seed_source = (AUDITED_SEED / 'summary_mean_H1_H3.csv'
                   if (AUDITED_SEED / 'summary_mean_H1_H3.csv').exists()
                   else SEED / 'summary_mean_H1_H3.csv')
    s = pd.read_csv(seed_source)
    CFG = [('CNN', 'CNN encoder'), ('LSTM', 'LSTM encoder'), ('CNN-LSTM', 'CNN--LSTM encoder'),
           ('M1', 'TCN, seasonal memory'), ('M2', 'M1 + climate context'), ('M3', 'M2 + commune conditioning'),
           ('M4', 'M3 + fixed graph'), ('M5', 'M3 + learned graph (neural count model)'),
           ('M6', 'M5 + dynamic spatial gate'), ('LSTM-FULL', 'LSTM, full inputs, no graph'),
           ('LSTM-GNN', 'LSTM + learned graph, gate'), ('CNN-LSTM-FULL', 'CNN--LSTM, full inputs, no graph'),
           ('CNN-LSTM-GNN', 'CNN--LSTM + learned graph, gate'), ('GNN', 'Last-state + learned graph, gate'),
           ('TCN-H64', 'M1, 64 hidden units'), ('TCN-L24', 'M1, 24-month history'),
           ('M5-ALERT', 'M5 + warning objective (outbreak-focused)')]
    wide = s.pivot(index='experiment', columns='metric', values=['mean', 'std'])
    out = []
    L = ['\\begin{table}[H]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}',
         '\\caption{Five-seed variability of all neural configurations on the 2021--2025 test period (mean H1--H3; seeds 9411--9415). '
         'Entries are mean $\\pm$ s.d. across seeds. Outbreak outcomes use the audited Ministry of Health definition; count-model '
         'probabilities are fractions of 500 predictive draws above that threshold, while M5-ALERT uses its prespecified auxiliary '
         'warning head. These descriptive test-period results were not used to choose the architecture, which was selected on the '
         '2019--2020 validation data.}\\label{tab:s_seed_all}',
         '\\resizebox{\\textwidth}{!}{\\begin{tabular}{llccccc}', '\\toprule', 'Code & Configuration & CRPS & Brier & Sensitivity & Specificity & F1\\\\', '\\midrule']
    for code, desc in CFG:
        cells = []
        for c, _, _ in MET:
            mu, sdv = wide.loc[code, ('mean', c)], wide.loc[code, ('std', c)]
            cells.append(pm(mu, sdv, 3))
            out.append(dict(experiment=code, configuration=desc, metric=c, mean=mu, sd=sdv, seeds=5))
        L.append(f'{code} & {desc} & ' + ' & '.join(cells) + '\\\\')
        if code in ('CNN-LSTM', 'M6', 'TCN-L24'):
            L.append('\\midrule')
    L += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
    write_tex('table_uncertainty_seeds.tex', '\n'.join(L) + '\n')
    pd.DataFrame(out).to_csv(SRC / 'Uncertainty_five_seed_all_configs.csv', index=False)
    print('uncertainty tables and source data written')


ARCH_ORDER = ['TCN-GNN', 'TCN-GNN + dynamic spatial gate', 'CNN-LSTM-GNN', 'LSTM-GNN', 'Last-state GNN']
FEAT_ORDER = ['Full set', 'Basic + soil moisture', 'Basic + dew point', 'Basic + solar radiation',
              'Basic + ET0', 'Basic climate']
ABL_MET = {'crps': 'crps', 'mae': 'mae', 'brier': 'outbreak_brier',
           'sensitivity': 'outbreak_sensitivity', 'f1': 'outbreak_f1'}


def ablations():
    """Five-seed validation ablations (trained through 2018, evaluated 2019-2020)."""
    d = pd.read_csv(UP / 'ablation_multiseed_validation' / 'figure7_mean_sd.csv')
    out = {}
    for kind, order, name, labcol, tex, cap, lab in [
        ('architecture', ARCH_ORDER, 'architecture', 'architecture', 'table_architecture.tex',
         'Architecture ablation on the 2019--2020 validation data (mean H1--H3). Entries are mean $\\pm$ s.d. over five training seeds (9411--9415); '
         'models were trained on 2010--2018 only.', 'tab:s_arch'),
        ('feature', FEAT_ORDER, 'feature_set', 'feature_set', 'table_features.tex',
         'Environmental feature ablation on the selected TCN--GNN backbone, 2019--2020 validation data (mean H1--H3). Entries are mean $\\pm$ s.d. '
         'over five training seeds; models were trained on 2010--2018 only.', 'tab:s_feat')]:
        sub = d[d.ablation == kind]
        rows = []
        for cfg in order:
            r = {labcol: cfg, 'seeds': 5}
            for k, m in ABL_MET.items():
                x = sub[(sub.configuration == cfg) & (sub.metric == m)].iloc[0]
                r[k], r[k + '_sd'] = x['mean'], x['sd']
            rows.append(r)
        df = pd.DataFrame(rows)
        for k in ('crps', 'f1'):
            best = (df[k].min() if k == 'crps' else df[k].max())
            df[k + '_gap_from_best_percent'] = 100 * (df[k] - best).abs() / best
        fn = 'Fig8_architecture_ablation.csv' if kind == 'architecture' else 'Fig8_feature_ablation.csv'
        df.to_csv(SRC / fn, index=False)
        df.to_csv(SRC / ('TableS1_architecture.csv' if kind == 'architecture' else 'TableS2_features.csv'), index=False)
        L = ['\\begin{table}[H]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}', f'\\caption{{{cap}}}\\label{{{lab}}}',
             '\\begin{tabular}{lccccc}', '\\toprule',
             ('Architecture' if kind == 'architecture' else 'Feature set') + ' & CRPS & MAE & Brier & Sensitivity & F1\\\\', '\\midrule']
        for _, r in df.iterrows():
            L.append(texify(r[labcol]) + ' & ' + ' & '.join(pm(r[k], r[k + '_sd'], 3) for k in ABL_MET) + '\\\\')
        L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
        write_tex(tex, '\n'.join(L) + '\n')


def support():
    d = pd.read_csv(UP / 'spatial_support_bootstrap' / 'support_block_bootstrap.csv')
    d.to_csv(SRC / 'Uncertainty_spatial_support_bootstrap.csv', index=False)
    lab = {'support_128': '128 units (median 4 communes)', 'support_32': '32 units (median 16)',
           'province_6': '6 units (median 102)'}
    mets = [('crps_incidence', 'CRPS'), ('mae_incidence', 'MAE'), ('brier_skill', 'Brier skill score')]
    L = ['\\begin{table}[H]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}',
         '\\caption{Controlled spatial aggregation: paired difference from commune support (coarser minus commune; mean H1--H3) with percentile 95\\% '
         'intervals from 2,000 target-calendar-month block-bootstrap replicates. The same commune-level predictive draws were summed, with no retraining. '
         'Negative CRPS and MAE differences and positive Brier-skill differences favour coarser support.}\\label{tab:s_support_boot}',
         '\\begin{tabular}{lccc}', '\\toprule', 'Prediction units & ' + ' & '.join(n for _, n in mets) + '\\\\', '\\midrule']
    for sc in ('support_128', 'support_32', 'province_6'):
        cells = []
        for m, _ in mets:
            r = d[(d.scale_a == sc) & (d.metric == m)].iloc[0]
            cells.append(f'{fnum(r.estimate)} ({fnum(r.ci_lower)}, {fnum(r.ci_upper)})')
        L.append(lab[sc] + ' & ' + ' & '.join(cells) + '\\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    write_tex('table_support_bootstrap.tex', '\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
    ablations()
    support()
    print('ablation and spatial-support uncertainty assets written')
