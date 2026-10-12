#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RES=ROOT/'results/current'; GEN=ROOT/'paper/generated'; SRC=ROOT/'source_data'
GEN.mkdir(parents=True,exist_ok=True); SRC.mkdir(parents=True,exist_ok=True)
mean=pd.read_csv(RES/'final_metrics_mean_H1_H3.csv')
by_horizon=pd.read_csv(RES/'final_metrics_by_horizon.csv')
later_mean=pd.read_csv(RES/'final_metrics_mean_H1_H3_2023_2025.csv')
arch=pd.read_csv(RES/'architecture_ablation.csv')
feat=pd.read_csv(RES/'feature_ablation.csv')
reported_mean=mean.loc[~mean.model.eq('Original ensemble')].copy()
reported_by_horizon=by_horizon.loc[~by_horizon.model.eq('Original ensemble')].copy()
reported_mean.to_csv(SRC/'Table1_model_summary.csv',index=False)
tradeoff=reported_mean.copy()
tradeoff['false_alert_rate']=1-tradeoff['outbreak_specificity']
tradeoff.to_csv(SRC/'Fig4_tradeoff.csv',index=False)
reported_by_horizon.to_csv(SRC/'Fig5_horizon_performance.csv',index=False)
operating=(reported_by_horizon.groupby('model',as_index=False)[
    ['outbreak_observed_positive','outbreak_tp','outbreak_fp','outbreak_tn']
].sum())
operating['detected_per_100_outbreaks']=100*operating.outbreak_tp/operating.outbreak_observed_positive
operating['false_alerts_per_100_nonoutbreaks']=100*operating.outbreak_fp/(operating.outbreak_fp+operating.outbreak_tn)
operating.to_csv(SRC/'Fig6_operating_point.csv',index=False)

hybrid='DeepST-INLA (proposed hybrid)'
full=mean.set_index('model').loc[hybrid]
later=later_mean.set_index('model').loc[hybrid]
rolling=pd.read_csv(RES/'rolling_refit_relative_change.csv').set_index('metric')['relative_change_percent']
temporal=[]
for label,column in [('MAE','mae'),('RMSE','rmse'),('CRPS','crps'),('Brier','outbreak_brier'),('F1','outbreak_f1')]:
    deterioration=float(-rolling[label] if label=='F1' else rolling[label])
    temporal.append({'metric':label,'full_2021_2025':full[column],
                     'later_2023_2025':later[column],
                     'later_to_full_ratio':later[column]/full[column],
                     'rolling_refit_deterioration_percent':deterioration})
pd.DataFrame(temporal).to_csv(SRC/'Fig7_temporal_robustness.csv',index=False)
pd.read_csv(RES/'rolling_refit_relative_change.csv').to_csv(SRC/'Fig7_rolling_refit.csv',index=False)
arch.to_csv(SRC/'TableS1_architecture.csv',index=False)
feat.to_csv(SRC/'TableS2_features.csv',index=False)

def latex_table(df, caption, label, cols, rename):
    x=df[cols].copy(); x=x.rename(columns=rename)
    for c in x.columns[1:]:
        if pd.api.types.is_numeric_dtype(x[c]):
            if c.lower() in {'n','seeds','target months','selected','outbreaks','captured','horizon'}:
                x[c]=x[c].map(lambda v:f'{int(v)}')
            else:
                x[c]=x[c].map(lambda v:f'{v:.3f}')
    body=x.to_latex(index=False,escape=True,column_format='l'+'r'*(len(x.columns)-1))
    if len(x.columns) >= 7:
        body='\\resizebox{\\textwidth}{!}{%\n'+body+'}%\n'
    return '\\begin{table}[H]\\centering\\small\n\\caption{'+caption+'}\\label{'+label+'}\n'+body+'\\end{table}\n'
(GEN/'table_architecture.tex').write_text(latex_table(arch,'Architecture ablation on validation data.','tab:s_arch',['architecture','crps','mae','brier','sensitivity','f1'],{'architecture':'Architecture','crps':'CRPS','mae':'MAE','brier':'Brier','sensitivity':'Sensitivity','f1':'F1'}))
(GEN/'table_features.tex').write_text(latex_table(feat,'Environmental feature ablation on the selected TCN--GNN backbone.','tab:s_feat',['feature_set','crps','mae','brier','sensitivity','f1'],{'feature_set':'Feature set','crps':'CRPS','mae':'MAE','brier':'Brier','sensitivity':'Sensitivity','f1':'F1'}))
full_metrics=reported_mean.loc[~reported_mean.model.eq('TCN-GNN Deep Ensemble (variant)')].copy()
(GEN/'table_full_metrics.tex').write_text(latex_table(full_metrics,'Mean H1--H3 performance on the held-out 2021--2025 test period.','tab:s_full',['model','mae','rmse','crps','outbreak_brier','outbreak_precision','outbreak_specificity','outbreak_sensitivity','outbreak_f1'],{'model':'Model','mae':'MAE','rmse':'RMSE','crps':'CRPS','outbreak_brier':'Brier','outbreak_precision':'Precision','outbreak_specificity':'Specificity','outbreak_sensitivity':'Sensitivity','outbreak_f1':'F1'}))

# Completed robustness analyses. Compact numerical outputs are copied into the
# source-data archive and rendered as supplementary tables when available.
baseline_frames=[]
for label,path in [
    ('Seasonal naive',ROOT/'results/seasonal_naive/metrics.csv'),
    ('Province allocation',ROOT/'results/province_allocation/metrics.csv'),
]:
    if path.exists():
        frame=pd.read_csv(path); frame['model']=label; baseline_frames.append(frame)
if baseline_frames:
    baselines=pd.concat(baseline_frames,ignore_index=True)
    baselines.to_csv(SRC/'TableS3_simple_baselines.csv',index=False)
    baseline_mean=baselines.groupby('model',as_index=False)[['crps','outbreak_brier','outbreak_sensitivity','outbreak_specificity','outbreak_f1']].mean()
    (GEN/'table_simple_baselines.tex').write_text(latex_table(baseline_mean,'Simple baselines on the exact locked 2021--2025 forecast keys.','tab:s_baselines',['model','crps','outbreak_brier','outbreak_sensitivity','outbreak_specificity','outbreak_f1'],{'model':'Model','crps':'CRPS','outbreak_brier':'Brier','outbreak_sensitivity':'Sensitivity','outbreak_specificity':'Specificity','outbreak_f1':'F1'}))

completed_tables=[
    ('bootstrap_pairwise.csv','TableS4_bootstrap.csv','table_bootstrap.tex','Mean H1--H3 paired target-month block-bootstrap contrasts (DeepST--INLA minus TCN--GNN; 2,000 replicates).','tab:s_bootstrap'),
    ('threshold_sensitivity.csv','TableS5_threshold_sensitivity.csv','table_threshold_sensitivity.tex','Outbreak-definition sensitivity with validation-only calibration and cutoff selection.','tab:s_threshold'),
    ('decision_utility_summary.csv','TableS6_decision_utility.csv','table_decision_utility.tex','Outbreak capture under monthly intervention-capacity constraints.','tab:s_utility'),
    ('multiseed_summary.csv','TableS7_multiseed.csv','table_multiseed.tex','Independent-seed variability for the principal neural paths.','tab:s_multiseed'),
    ('spatial_holdout_summary.csv','TableS8_spatial_holdout.csv','table_spatial_holdout.tex','Combined predictions from six leave-one-parent-area-out TCN--GNN folds.','tab:s_spatial'),
    ('timesfm3_summary.csv','TableS9_timesfm3.csv','table_timesfm3.tex','TimesFM-3 and frozen TimesFM-3--GNN comparison.','tab:s_timesfm'),
]
for filename,source_name,tex_name,caption,label in completed_tables:
    path=ROOT/'results/pending'/filename
    if not path.exists(): continue
    frame=pd.read_csv(path); frame.to_csv(SRC/source_name,index=False)
    if filename == 'bootstrap_pairwise.csv':
        display=frame[frame.scope.eq('mean_H1_H3')].copy()
        columns=['metric','estimate_a','estimate_b','difference_a_minus_b','ci025','ci975']
        rename={'metric':'Metric','estimate_a':'DeepST--INLA','estimate_b':'TCN--GNN','difference_a_minus_b':'Difference','ci025':'95\\% CI lower','ci975':'95\\% CI upper'}
    elif filename == 'threshold_sensitivity.csv':
        display=frame.groupby('definition',as_index=False)[['test_brier','test_sensitivity','test_specificity','test_f1']].mean()
        order=['mean_plus_1.5sd','moh_audited','mean_plus_2.5sd','percentile_90']
        display['definition']=pd.Categorical(display.definition,categories=order,ordered=True)
        display=display.sort_values('definition')
        display['definition']=display.definition.astype(str).map({'mean_plus_1.5sd':'Mean + 1.5 s.d.','moh_audited':'Audited MOH','mean_plus_2.5sd':'Mean + 2.5 s.d.','percentile_90':'Rolling 90th percentile'})
        columns=['definition','test_brier','test_sensitivity','test_specificity','test_f1']
        rename={'definition':'Definition','test_brier':'Brier','test_sensitivity':'Sensitivity','test_specificity':'Specificity','test_f1':'F1'}
    elif filename == 'decision_utility_summary.csv':
        display=frame.copy()
        columns=['horizon','budget','pooled_capture_rate','selection_precision','capture_lift_over_random']
        rename={'horizon':'Horizon','budget':'Capacity','pooled_capture_rate':'Outbreak capture','selection_precision':'Selection precision','capture_lift_over_random':'Lift over random'}
    elif filename == 'multiseed_summary.csv':
        display=frame[frame.scope.eq('mean_H1_H3') & frame.metric.isin(['crps','outbreak_brier','outbreak_f1'])].copy()
        display['family']=display.family.map({'base':'TCN--GNN','alert':'Outbreak-focused'})
        display['metric']=display.metric.map({'crps':'CRPS','outbreak_brier':'Brier','outbreak_f1':'F1'})
        columns=['family','metric','mean','std','seeds']
        rename={'family':'Family','metric':'Metric','mean':'Mean','std':'S.d.','seeds':'Seeds'}
    elif filename == 'spatial_holdout_summary.csv':
        display=frame.copy()
        columns=['horizon','n','crps','outbreak_brier','outbreak_sensitivity','outbreak_specificity','outbreak_f1']
        rename={'horizon':'Horizon','n':'N','crps':'CRPS','outbreak_brier':'Brier','outbreak_sensitivity':'Sensitivity','outbreak_specificity':'Specificity','outbreak_f1':'F1'}
    elif filename == 'timesfm3_summary.csv':
        display=frame.copy()
        columns=['model','horizon','crps','outbreak_brier','outbreak_sensitivity','outbreak_specificity','outbreak_f1']
        rename={'model':'Model','horizon':'Horizon','crps':'CRPS','outbreak_brier':'Brier','outbreak_sensitivity':'Sensitivity','outbreak_specificity':'Specificity','outbreak_f1':'F1'}
    else:
        display=frame; columns=list(frame.columns)
        rename={c:c.replace('_',' ').title() for c in columns}
    (GEN/tex_name).write_text(latex_table(display,caption,label,columns,rename))

print('Generated source data and LaTeX tables.')
