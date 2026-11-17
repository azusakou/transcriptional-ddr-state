#!/usr/bin/env python3
from pathlib import Path
import os
import numpy as np,pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error
R=Path(os.environ['DDR_EXTENSION_OUTPUT']).resolve()
d=pd.read_csv(R/'curated_all_outer_predictions.csv',float_precision='round_trip');rng=np.random.default_rng(6402);out=[]
new=list(d[d.model.str.contains('__cross_entropy|__mse|selector_curated')].model.unique())
main='state_curated_TP53_Gem_pharmacology__cross_entropy'
refs=['parsimonious_additive','drug_only','ddr_fraction_additive','curated_TP53_all_regimens__cross_entropy','state_curated_TP53_all_regimens__cross_entropy','curated_TP53_Gem_pharmacology_only__cross_entropy','mechanistic_TP53']
keep=list(dict.fromkeys(new+refs));pairs_to_compare=[(m,'parsimonious_additive') for m in new]+[(main,r) for r in refs if r!='parsimonious_additive']
for drug in ['all','Gemcitabine','Idasanutlin','Prexasertib','Bortezomib']:
 z=d if drug=='all' else d[d.drug.eq(drug)];base=z[z.model.eq('drug_only')].set_index(['cell_line','regimen']).sort_index();y=base.observed.to_numpy();line=base.index.get_level_values(0).to_numpy();lines=np.array(sorted(set(line)));blocks={x:np.flatnonzero(line==x) for x in lines}
 pm={m:z[z.model.eq(m)].set_index(['cell_line','regimen']).loc[base.index].predicted.to_numpy() for m in keep};vals={m:[] for m in keep};deltas={k:[] for k in pairs_to_compare}
 for b in range(4000):
  ix=np.concatenate([blocks[x] for x in rng.choice(lines,len(lines),replace=True)]);yy=y[ix];den=np.sum((yy-yy.mean())**2);met={m:[1-np.sum((yy-p[ix])**2)/den,np.mean(abs(yy-p[ix]))] for m,p in pm.items()}
  for m in vals:vals[m].append(met[m])
  for m,r in deltas:deltas[(m,r)].append([met[m][0]-met[r][0],met[r][1]-met[m][1]])
 for m,a in vals.items():
  for j,st in enumerate(['r2','mae']):
   lo,hi=np.quantile(np.asarray(a)[:,j],[.025,.975]);est=r2_score(y,pm[m]) if j==0 else mean_absolute_error(y,pm[m]);out.append(dict(drug=drug,model=m,reference_model='none',statistic=st,estimate=est,ci_low=lo,ci_high=hi,n_lines=len(lines),n_bootstrap=4000))
 for (m,r),a in deltas.items():
  for j,st in enumerate(['delta_r2','mae_improvement']):
   lo,hi=np.quantile(np.asarray(a)[:,j],[.025,.975]);est=r2_score(y,pm[m])-r2_score(y,pm[r]) if j==0 else mean_absolute_error(y,pm[r])-mean_absolute_error(y,pm[m]);out.append(dict(drug=drug,model=m,reference_model=r,statistic=st,estimate=est,ci_low=lo,ci_high=hi,n_lines=len(lines),n_bootstrap=4000))
pd.DataFrame(out).to_csv(R/'curated_biology_line_bootstrap.csv',index=False)
print(pd.DataFrame(out).query("model==@main and (drug=='all' or (drug=='Gemcitabine' and (reference_model=='none' or reference_model=='parsimonious_additive')))").to_string(index=False))
