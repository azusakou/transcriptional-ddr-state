#!/usr/bin/env python3
from pathlib import Path
import os
import sys,json
import numpy as np,pandas as pd
from scipy.stats import rankdata,spearmanr
from sklearn.metrics import r2_score,mean_absolute_error
import evaluate_extensions as ev
R=Path(os.environ['DDR_EXTENSION_OUTPUT']).resolve()
sys.path.insert(0,str(Path(os.environ['DDR_EXTENSION_METHODS'])/'context'))
from reanalyse_functional_context import correlation,bh
d=pd.read_csv(R/'all_outer_predictions.csv',float_precision='round_trip')
models=list(d.model.unique());rng=np.random.default_rng(6402);rows=[]
for drug in ['all']+ev.core.DRUGS:
 q=d if drug=='all' else d[d.drug.eq(drug)]
 base=q[q.model.eq('drug_only')].set_index(['cell_line','regimen']).sort_index();y=base.observed.to_numpy();line=base.index.get_level_values(0).to_numpy();ul=np.array(sorted(set(line)));blocks={a:np.flatnonzero(line==a) for a in ul}
 pm={m:q[q.model.eq(m)].set_index(['cell_line','regimen']).loc[base.index].predicted.to_numpy() for m in models};vals={m:[] for m in models};diff={m:[] for m in models if m!='parsimonious_additive'}
 for rep in range(4000):
  ix=np.concatenate([blocks[a] for a in rng.choice(ul,len(ul),replace=True)]);yy=y[ix];den=np.sum((yy-yy.mean())**2)
  mm={m:[1-np.sum((yy-p[ix])**2)/den,np.mean(abs(yy-p[ix]))] for m,p in pm.items()}
  for m in vals:vals[m].append(mm[m])
  for m in diff:diff[m].append([mm[m][0]-mm['parsimonious_additive'][0],mm['parsimonious_additive'][1]-mm[m][1]])
 for m,a in vals.items():
  for j,st in enumerate(['r2','mae']):
   lo,hi=np.quantile(np.asarray(a)[:,j],[.025,.975]);estimate=r2_score(y,pm[m]) if j==0 else mean_absolute_error(y,pm[m]);rows.append(dict(drug=drug,model=m,reference_model='none',statistic=st,estimate=estimate,ci_low=lo,ci_high=hi,n_lines=len(ul),n_bootstrap=4000))
 for m,a in diff.items():
  for j,st in enumerate(['delta_r2','mae_improvement']):
   lo,hi=np.quantile(np.asarray(a)[:,j],[.025,.975]);estimate=r2_score(y,pm[m])-r2_score(y,pm['parsimonious_additive']) if j==0 else mean_absolute_error(y,pm['parsimonious_additive'])-mean_absolute_error(y,pm[m]);rows.append(dict(drug=drug,model=m,reference_model='parsimonious_additive',statistic=st,estimate=estimate,ci_low=lo,ci_high=hi,n_lines=len(ul),n_bootstrap=4000))
pd.DataFrame(rows).to_csv(R/'all_line_bootstrap.csv',index=False)

geno=pd.read_csv(R/'source_TP53_annotations.csv');geno['TP53_group']=np.where((geno.TP53_DAM_MUT==0)&(geno.TP53_DAM_HOT==0),'no_damaging_or_hotspot_annotation','damaging_or_hotspot_annotation')
ida=d[d.drug.eq('Idasanutlin')].merge(geno[['cell_line','TP53_group']],on='cell_line',validate='many_to_one');rows=[]
for (m,group),a in ida.groupby(['model','TP53_group']):
 rows.append(dict(model=m,TP53_group=group,n_pairs=len(a),n_lines=a.cell_line.nunique(),r2=r2_score(a.observed,a.predicted),mae=mean_absolute_error(a.observed,a.predicted),rmse=float(np.sqrt(np.mean((a.observed-a.predicted)**2))),mean_observed=a.observed.mean(),mean_predicted=a.predicted.mean()))
pd.DataFrame(rows).to_csv(R/'idasanutlin_TP53_stratum_errors.csv',index=False)

# Join one source viability value only after equal averaging of represented exposure times.
source=pd.read_csv(Path(os.environ['DDR_EXTENSION_CONTEXT_INPUTS'])/'frozen_forecasts_with_PRISM.csv')
source=source[['cell_line','drug','PRISM_AUC']].drop_duplicates()
assert not source.duplicated(['cell_line','drug']).any()
agg=d.groupby(['cell_line','drug','model']).agg(observed=('observed','mean'),predicted=('predicted','mean'),n_regimens=('source_time','size')).reset_index()
joined=agg.merge(source,on=['cell_line','drug'],validate='many_to_one').merge(geno[['cell_line','TP53_group']],on='cell_line',how='left',validate='many_to_one')
joined.to_csv(R/'extended_predictions_with_PRISM.csv',index=False)
rows=[]
for (m,drug),a in joined.groupby(['model','drug']):
 if m not in ['mechanistic_TP53','TP53_only','parsimonious_additive','nested_selector','selector_augmented_mse','transcriptome_PC_cross_entropy']:continue
 z=correlation(a.predicted.to_numpy(),a.PRISM_AUC.to_numpy());rows.append(dict(model=m,drug=drug,**z))
out=pd.DataFrame(rows);out['q_four_drugs']=np.nan
for m,a in out.groupby('model'):out.loc[a.index,'q_four_drugs']=bh(a.p_permutation.to_numpy())
out['q_all_24_context_tests']=bh(out.p_permutation.to_numpy())
out.to_csv(R/'extended_PRISM_associations.csv',index=False)
rows=[]
for m in ['mechanistic_TP53','TP53_only','parsimonious_additive','nested_selector','transcriptome_PC_cross_entropy']:
 a=joined[(joined.model==m)&(joined.drug=='Idasanutlin')&joined.PRISM_AUC.notna()].copy()
 for label,z in [('all',a)]+list(a.groupby('TP53_group')):
  rows.append(dict(model=m,group=label,n_lines=len(z),rho_with_PRISM_AUC=spearmanr(z.predicted,z.PRISM_AUC).statistic))
 Z=np.column_stack([np.ones(len(a)),a.TP53_group.eq('damaging_or_hotspot_annotation').astype(int)]);x=rankdata(a.predicted);y=rankdata(a.PRISM_AUC);rx=x-Z@np.linalg.lstsq(Z,x,rcond=None)[0];ry=y-Z@np.linalg.lstsq(Z,y,rcond=None)[0]
 rows.append(dict(model=m,group='rank residual adjusted binary TP53 annotation',n_lines=len(a),rho_with_PRISM_AUC=float(np.corrcoef(rx,ry)[0,1])))
pd.DataFrame(rows).to_csv(R/'idasanutlin_extended_PRISM_by_TP53.csv',index=False)
print('ALL intervals/context complete',flush=True)
print(out[out.model.eq('mechanistic_TP53')].to_string(index=False),flush=True)
print(pd.DataFrame(rows).query("model=='mechanistic_TP53'").to_string(index=False),flush=True)
