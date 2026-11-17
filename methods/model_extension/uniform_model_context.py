#!/usr/bin/env python3
from pathlib import Path
import os
import sys
import numpy as np,pandas as pd
from scipy.stats import rankdata,spearmanr
from sklearn.metrics import r2_score,mean_absolute_error
import evaluate_extensions as ev
R=Path(os.environ['DDR_EXTENSION_OUTPUT']).resolve()
sys.path.insert(0,str(Path(os.environ['DDR_EXTENSION_METHODS'])/'context'))
from reanalyse_functional_context import correlation,bh
d=pd.read_csv(R/'uniform_all_outer_predictions.csv',float_precision='round_trip');geno=pd.read_csv(R/'uniform_TP53_annotations.csv')
geno['TP53_call_group']=np.where(geno.TP53_protein_altering.eq(1),'qualifying_protein_altering_call','profiled_no_qualifying_call')
main='state_CCLE_TP53_Gem_pharmacology__cross_entropy';families=['CCLE_TP53_all_regimens__cross_entropy','state_CCLE_TP53_all_regimens__cross_entropy',main,'CCLE_TP53_Gem_pharmacology_only__cross_entropy','selector_uniform_cross_entropy','selector_uniform_mse']
records=[]
for (m,drug),a in d[d.model.isin(families)].groupby(['model','drug']):
 slope,intercept=np.polyfit(a.predicted,a.observed,1);records.append(dict(model=m,drug=drug,n_lines=a.cell_line.nunique(),**ev.core.metrics(a.observed.to_numpy(),a.predicted.to_numpy()),calibration_intercept=intercept,calibration_slope=slope))
for m,a in d[d.model.isin(families)].groupby('model'):
 slope,intercept=np.polyfit(a.predicted,a.observed,1);q=ev.core.metrics(a.observed.to_numpy(),a.predicted.to_numpy());records.append(dict(model=m,drug='all',n_lines=a.cell_line.nunique(),**q,calibration_intercept=intercept,calibration_slope=slope))
pd.DataFrame(records).to_csv(R/'uniform_model_calibration.csv',index=False)
selected=d[d.model.isin(families+['parsimonious_additive','mechanistic_TP53'])].merge(geno[['cell_line','TP53_call_group']],on='cell_line',validate='many_to_one')
records=[]
for (m,drug,group),a in selected.groupby(['model','drug','TP53_call_group']):
 records.append(dict(model=m,drug=drug,TP53_call_group=group,n_pairs=len(a),n_lines=a.cell_line.nunique(),r2=r2_score(a.observed,a.predicted) if len(a)>1 else np.nan,mae=mean_absolute_error(a.observed,a.predicted),rmse=float(np.sqrt(np.mean((a.observed-a.predicted)**2))),mean_observed=a.observed.mean(),mean_predicted=a.predicted.mean()))
pd.DataFrame(records).to_csv(R/'uniform_TP53_stratum_errors.csv',index=False)
source=pd.read_csv(Path(os.environ['DDR_EXTENSION_CONTEXT_INPUTS'])/'frozen_forecasts_with_PRISM.csv')[['cell_line','drug','PRISM_AUC']].drop_duplicates();assert not source.duplicated(['cell_line','drug']).any()
agg=d[d.model.isin(families)].groupby(['cell_line','drug','model']).agg(observed=('observed','mean'),predicted=('predicted','mean'),n_regimens=('source_time','size')).reset_index().merge(source,on=['cell_line','drug'],validate='many_to_one').merge(geno[['cell_line','TP53_call_group']],on='cell_line',validate='many_to_one')
agg.to_csv(R/'uniform_predictions_with_PRISM.csv',index=False);records=[]
for (m,drug),a in agg.groupby(['model','drug']):records.append(dict(model=m,drug=drug,**correlation(a.predicted.to_numpy(),a.PRISM_AUC.to_numpy())))
res=pd.DataFrame(records);res['q_four_drugs']=np.nan
for m,a in res.groupby('model'):res.loc[a.index,'q_four_drugs']=bh(a.p_permutation)
res['q_all24_uniform_context_tests']=bh(res.p_permutation);res.to_csv(R/'uniform_PRISM_associations.csv',index=False)
records=[]
for (m,drug),a in agg.groupby(['model','drug']):
 a=a[a.PRISM_AUC.notna()].copy();z=np.column_stack([np.ones(len(a)),a.TP53_call_group.eq('qualifying_protein_altering_call').astype(int)]);x=rankdata(a.predicted);y=rankdata(a.PRISM_AUC);rx=x-z@np.linalg.lstsq(z,x,rcond=None)[0];ry=y-z@np.linalg.lstsq(z,y,rcond=None)[0]
 records.append(dict(model=m,drug=drug,group='rank residual adjusted binary TP53 call',n_lines=len(a),rho_with_PRISM_AUC=np.corrcoef(rx,ry)[0,1]))
 for group,h in a.groupby('TP53_call_group'):records.append(dict(model=m,drug=drug,group=group,n_lines=len(h),rho_with_PRISM_AUC=spearmanr(h.predicted,h.PRISM_AUC).statistic))
pd.DataFrame(records).to_csv(R/'uniform_PRISM_TP53_strata.csv',index=False)
print(pd.DataFrame(records).query('model==@main').to_string(index=False))
