from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error

R=Path(os.environ['DDR_EXTENSION_OUTPUT']).resolve()
d=pd.read_csv(R/'outer_predictions.csv',float_precision='round_trip')
rng=np.random.default_rng(6402);out=[]
for drug in ['all','Gemcitabine','Idasanutlin','Prexasertib','Bortezomib']:
 z=d if drug=='all' else d[d.drug.eq(drug)]
 main=z[z.model.eq('mechanistic_TP53')].set_index(['cell_line','regimen']).sort_index()
 y=main.observed.to_numpy();line=main.index.get_level_values(0).to_numpy();lines=np.array(sorted(set(line)))
 blocks={x:np.flatnonzero(line==x) for x in lines}
 preds={m:z[z.model.eq(m)].set_index(['cell_line','regimen']).loc[main.index].predicted.to_numpy() for m in ['mechanistic_TP53','TP53_only','drug_only','ddr_fraction_additive','parsimonious_additive']}
 vals={m:[] for m in preds};deltas={m:[] for m in preds if m!='mechanistic_TP53'}
 for b in range(4000):
  ix=np.concatenate([blocks[x] for x in rng.choice(lines,len(lines),replace=True)]);v=y[ix];den=np.sum((v-v.mean())**2)
  met={m:[1-np.sum((v-p[ix])**2)/den,np.mean(abs(v-p[ix]))] for m,p in preds.items()}
  for m in vals:vals[m].append(met[m])
  for m in deltas:deltas[m].append([met['mechanistic_TP53'][0]-met[m][0],met[m][1]-met['mechanistic_TP53'][1]])
 for m,a in vals.items():
  for j,stat in enumerate(['r2','mae']):
   low,hi=np.quantile(np.array(a)[:,j],[.025,.975]);est=r2_score(y,preds[m]) if j==0 else mean_absolute_error(y,preds[m]);out.append(dict(drug=drug,model=m,reference_model='none',statistic=stat,estimate=est,ci_low=low,ci_high=hi,n_lines=len(lines),n_bootstrap=4000))
 for m,a in deltas.items():
  for j,stat in enumerate(['delta_r2','mae_improvement']):
   low,hi=np.quantile(np.array(a)[:,j],[.025,.975]);est=r2_score(y,preds['mechanistic_TP53'])-r2_score(y,preds[m]) if j==0 else mean_absolute_error(y,preds[m])-mean_absolute_error(y,preds['mechanistic_TP53']);out.append(dict(drug=drug,model='mechanistic_TP53',reference_model=m,statistic=stat,estimate=est,ci_low=low,ci_high=hi,n_lines=len(lines),n_bootstrap=4000))
pd.DataFrame(out).to_csv(R/'mechanistic_line_bootstrap.csv',index=False)
print(pd.DataFrame(out).query("drug in ['all','Idasanutlin'] and (model=='mechanistic_TP53' or (drug=='Idasanutlin' and statistic=='r2'))").to_string(index=False))

# Normalized errors and count strata are descriptive; they do not change target/metric.
base=d[d.model.eq('parsimonious_additive')].copy();nr=[]
for drug,g in base.groupby('drug'):
 sd=float(g.observed.std(ddof=0));iqr=float(np.subtract(*np.quantile(g.observed,[.75,.25])))
 nr.append(dict(drug=drug,n_pairs=len(g),n_lines=g.cell_line.nunique(),target_sd=sd,target_iqr=iqr,rmse=float(np.sqrt(np.mean((g.observed-g.predicted)**2))),mae=float(np.mean(abs(g.observed-g.predicted))),rmse_over_sd=float(np.sqrt(np.mean((g.observed-g.predicted)**2))/sd),median_treated_cells=float(g.treated_n_cells.median())))
pd.DataFrame(nr).to_csv(R/'within_drug_normalized_errors.csv',index=False)

# Verify the mechanistic conclusion is not driven by a single line's two times.
z=d[d.drug.eq('Idasanutlin')].copy();main=z[z.model.eq('mechanistic_TP53')].set_index(['cell_line','regimen']).sort_index();p0=z[z.model.eq('parsimonious_additive')].set_index(['cell_line','regimen']).loc[main.index].predicted.to_numpy();pm=main.predicted.to_numpy();y=main.observed.to_numpy();line=main.index.get_level_values(0).to_numpy();checks=[]
for x in sorted(set(line)):
 ix=line!=x;checks.append(dict(omitted_line=x,n_pairs=int(ix.sum()),mechanistic_r2=r2_score(y[ix],pm[ix]),delta_r2_vs_original=r2_score(y[ix],pm[ix])-r2_score(y[ix],p0[ix]),mae_gain_vs_original=mean_absolute_error(y[ix],p0[ix])-mean_absolute_error(y[ix],pm[ix])))
pd.DataFrame(checks).to_csv(R/'idasanutlin_line_influence_sensitivity.csv',index=False)
print('Idasanutlin omit-one-line deltaR2 range',pd.DataFrame(checks).delta_r2_vs_original.min(),pd.DataFrame(checks).delta_r2_vs_original.max())
