#!/usr/bin/env python3
from pathlib import Path
import sys,json,hashlib,itertools,time,os
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score,mean_absolute_error

ROOT=Path(os.environ.get('DDR_EXTENSION_PROJECT', '.'))
HERE=Path(os.environ.get('DDR_EXTENSION_OUTPUT', '.')).resolve()
sys.path.insert(0,os.environ['DDR_EXTENSION_METHODS'])
import source_matched_forecasting as sm
import nested_forecasting as core

INPUT=Path(os.environ['DDR_EXTENSION_FORECAST_INPUTS'])
OUT=HERE
ALPHAS={
 'parsimonious_additive':[.3,3,30],
 'programmes_additive':[.3,3,30,300],
 'distribution_additive':[.3,3,30,300],
 'TP53_only':[.3,3,30],
 'mechanistic_TP53':[.3,3,30],
 'ddr_fraction_additive':[.3,3,30],
 'drug_only':[0],
}
FAMILIES=['parsimonious_additive','TP53_only','mechanistic_TP53','distribution_additive','programmes_additive','shallow_boosting']

def loss(y,p):
 p=np.clip(p,1e-8,1-1e-8)
 return -(y*np.log(p)+(1-y)*np.log1p(-p))

def load():
 meta=pd.read_csv(INPUT/'cells.csv.gz',dtype={'source_time':str,'time':str},low_memory=False)
 names=json.loads((INPUT/'input_manifest.json').read_text())['module_column_order']
 modules={n:[] for n in names}
 scores=np.load(INPUT/'module_scores.npz')['scores']
 pairs,controls,treated=sm.cohort(meta,meta,True,False)
 foldmap=pd.read_csv(INPUT/'frozen_line_folds.csv').set_index('cell_line').fold.to_dict()
 folds=pairs.cell_line.map(foldmap).to_numpy(int)
 assert len(pairs)==243 and pairs.cell_line.nunique()==84
 regimes=list(dict.fromkeys(pairs.drug))
 D=np.column_stack([pairs.drug.eq(d) for d in regimes]).astype(float)
 genotype=pd.read_csv(Path(os.environ['DDR_EXTENSION_CONTEXT_INPUTS'])/'Idasanutlin_24hr_expt1_CL_features.csv')
 genotype['cell_line']=genotype.CCLE_ID.str.split('_').str[0]
 assert genotype.cell_line.is_unique
 annot=genotype.set_index('cell_line').TP53_DAM_HOT.to_dict()
 isida=pairs.compound.eq('Idasanutlin')
 assert pairs.loc[isida,'cell_line'].isin(annot).all()
 nohot=np.array([float(annot.get(c,1)==0) for c in pairs.cell_line])
 G=np.column_stack([nohot*pairs.drug.eq('Idasanutlin_6h'),nohot*pairs.drug.eq('Idasanutlin_24h')]).astype(float)
 pairs['TP53_annotation_available']=pairs.cell_line.isin(annot)
 pairs['TP53_without_damaging_hotspot']=nohot
 pairs.rename(columns={'drug':'regimen','compound':'drug'}).to_csv(OUT/'cohort.csv',index=False)
 genotype[['cell_line','TP53_DAM_MUT','TP53_DAM_HOT','CCLE_ID']].to_csv(OUT/'source_TP53_annotations.csv',index=False)
 return meta,names,modules,scores,pairs,controls,treated,folds,regimes,D,G

def features(scores,names,modules,pairs,controls,treated,trainlines,G):
 B,y,ref=sm.transformer_for(modules,'ddr_composite',2,False)(scores,pairs,controls,treated,trainlines)
 center=np.array([ref['score_centers'][n] for n in names]);scale=np.array([ref['score_scales'][n] for n in names]);z=(scores-center)/scale
 means=np.vstack([z[controls[(c,d)]].mean(0) for c,d in zip(pairs.cell_line,pairs.drug)])
 jddr,jp53,jcy,js=[names.index(n) for n in ['ddr_composite','TP53_response','cycle','stress']]
 tails=[]
 for c,d in zip(pairs.cell_line,pairs.drug):
  a=z[controls[(c,d)]]
  tails.append([a[:,jddr].std(),np.quantile(a[:,jddr],.9),a[:,jp53].std(),np.quantile(a[:,jp53],.9),(a[:,jcy]>2).mean(),np.quantile(a[:,js],.9)])
 T=np.asarray(tails)
 programs=np.column_stack([B[:,0],means])
 F={'drug_only':np.empty((len(B),0)),
    'ddr_fraction_additive':B[:,:1],
    'parsimonious_additive':B,
    'programmes_additive':programs,
    'distribution_additive':np.column_stack([B,T]),
    'TP53_only':G,
    'mechanistic_TP53':np.column_stack([B,G]),
    'shallow_boosting':np.column_stack([programs,G])}
 return F,y,ref

def linear_predict(B,D,y,tr,te,alpha):
 cen=B[tr].mean(0);sc=B[tr].std(0);sc[sc<1e-12]=1
 X=np.column_stack([D,(B-cen)/sc]);pen=np.r_[np.zeros(D.shape[1]),np.ones(B.shape[1])]
 beta,ok,nit=core.fit_model(X[tr],y[tr],pen,alpha)
 return expit(X[te]@beta),dict(alpha=alpha,feature_center=cen.tolist(),feature_scale=sc.tolist(),coefficients=beta.tolist(),converged=bool(ok),iterations=int(nit))

def boost_predict(B,D,y,tr,te,spec):
 x=np.column_stack([D,B]);m=HistGradientBoostingRegressor(loss='squared_error',max_iter=200,learning_rate=.05,early_stopping=False,random_state=20261117,**spec)
 m.fit(x[tr],y[tr]);raw=m.predict(x[te]);p=np.clip(raw,0,1)
 return p,dict(**spec,n_unconstrained_out_of_bounds=int(((raw<0)|(raw>1)).sum()))

def calculate(pred):
 result=[]
 for (model,drug),g in pred.groupby(['model','drug']):
  result.append(dict(model=model,drug=drug,n_lines=g.cell_line.nunique(),**core.metrics(g.observed.to_numpy(),g.predicted.to_numpy())))
 for model,g in pred.groupby('model'):
  result.append(dict(model=model,drug='all',n_lines=g.cell_line.nunique(),**core.metrics(g.observed.to_numpy(),g.predicted.to_numpy())))
 return pd.DataFrame(result)

def diagnostics(pred):
 g=pred[pred.model=='parsimonious_additive'].copy()
 g['residual']=g.observed-g.predicted;g['squared_error']=g.residual**2;g['absolute_error']=abs(g.residual)
 # Independent-cell binomial sampling is a descriptive reference, not biological-replicate uncertainty.
 g['iid_binomial_variance_unbiased']=g.observed*(1-g.observed)/(g.treated_n_cells-1)
 g.to_csv(OUT/'original_residuals.csv',index=False)
 rows=[]
 for drug,a in [('all',g)]+[(d,z) for d,z in g.groupby('drug')]:
  totvar=float(np.var(a.observed));noise=float(a.iid_binomial_variance_unbiased.mean());mse=float(a.squared_error.mean())
  rows.append(dict(drug=drug,n=len(a),observed_variance=totvar,mean_iid_binomial_variance=noise,iid_binomial_reference_R2_max=1-noise/totvar,observed_mse=mse,noise_fraction_of_mse=noise/mse,count_absolute_error_spearman=spearmanr(a.treated_n_cells,a.absolute_error).statistic))
 pd.DataFrame(rows).to_csv(OUT/'sampling_noise_reference.csv',index=False)
 inf=[]
 for line in sorted(g.cell_line.unique()):
  a=g[g.cell_line.ne(line)];base=r2_score(a.observed,a.predicted)
  z=g[g.cell_line.eq(line)];inf.append(dict(cell_line=line,n_pairs=len(z),sum_squared_error=z.squared_error.sum(),mean_absolute_error=z.absolute_error.mean(),leave_line_out_r2=base))
 pd.DataFrame(inf).sort_values('sum_squared_error',ascending=False).to_csv(OUT/'line_residual_influence.csv',index=False)

def bootstrap(pred):
 rng=np.random.default_rng(6402);rows=[]
 for drug in ['all']+core.DRUGS:
  q=pred if drug=='all' else pred[pred.drug==drug]
  models=list(q.model.unique());g=q[q.model=='drug_only'].copy();keys=['cell_line','regimen'];base=g.set_index(keys).sort_index();y=base.observed.to_numpy();line=base.index.get_level_values(0).to_numpy();ul=np.array(sorted(set(line)));blocks={k:np.flatnonzero(line==k) for k in ul}
  pm={m:q[q.model==m].set_index(keys).loc[base.index].predicted.to_numpy() for m in models}
  stats={m:[] for m in models};comparisons={m:[] for m in ['drug_only','ddr_fraction_additive','parsimonious_additive','TP53_only']}
  for k in range(4000):
   ix=np.concatenate([blocks[x] for x in rng.choice(ul,len(ul),replace=True)]);yy=y[ix];den=np.sum((yy-yy.mean())**2)
   vals={m:[1-np.sum((yy-p[ix])**2)/den,np.mean(abs(yy-p[ix]))] for m,p in pm.items()}
   for m in models:stats[m].append(vals[m])
   for m in comparisons:comparisons[m].append([vals['nested_selector'][0]-vals[m][0],vals[m][1]-vals['nested_selector'][1]])
  for m,a in stats.items():
   for j,st in enumerate(['r2','mae']):
    point=core.metrics(y,pm[m])[st];low,high=np.quantile(np.asarray(a)[:,j],[.025,.975]);rows.append(dict(drug=drug,model=m,reference_model='none',statistic=st,estimate=point,ci_low=low,ci_high=high,n_lines=len(ul),n_bootstrap=4000))
  for ref,a in comparisons.items():
   for j,st in enumerate(['delta_r2','mae_improvement']):
    point=(r2_score(y,pm['nested_selector'])-r2_score(y,pm[ref])) if j==0 else (np.mean(abs(y-pm[ref]))-np.mean(abs(y-pm['nested_selector'])))
    low,high=np.quantile(np.asarray(a)[:,j],[.025,.975]);rows.append(dict(drug=drug,model='nested_selector',reference_model=ref,statistic=st,estimate=point,ci_low=low,ci_high=high,n_lines=len(ul),n_bootstrap=4000))
 pd.DataFrame(rows).to_csv(OUT/'line_bootstrap.csv',index=False)

def main():
 started=time.time();plan=(OUT/'extension_plan.json').read_bytes();print('FROZEN PLAN',hashlib.sha256(plan).hexdigest(),flush=True)
 meta,names,modules,scores,pairs,controls,treated,folds,regimes,D,G=load()
 tuning=[];pred=[];pars=[];refs=[];choices=[]
 boost_specs=[dict(max_leaf_nodes=l,min_samples_leaf=n,l2_regularization=a) for l,n,a in itertools.product([3,5],[8,16],[3,30])]
 for fold in sorted(set(folds)):
  tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold)
  assert not set(pairs.iloc[tr].cell_line)&set(pairs.iloc[te].cell_line)
  outerF,y,ref=features(scores,names,modules,pairs,controls,treated,set(pairs.iloc[tr].cell_line),G)
  refs.append(dict(level='outer',fold=int(fold),**ref))
  inner=[]
  for innerfold,(itr,iva) in enumerate(GroupKFold(3).split(tr,groups=pairs.iloc[tr].cell_line),1):
   ii=tr[itr];vv=tr[iva];FF,yy,rr=features(scores,names,modules,pairs,controls,treated,set(pairs.iloc[ii].cell_line),G)
   refs.append(dict(level='inner',fold=int(fold),inner_fold=innerfold,**rr));inner.append((ii,vv,FF,yy))
  best={}
  for family,alphas in ALPHAS.items():
   trials=[]
   for alpha in alphas:
    ls=[]
    for ii,vv,FF,yy in inner:
     pp,par=linear_predict(FF[family],D,yy,ii,vv,alpha);ls.extend(loss(yy[vv],pp))
    mean=float(np.mean(ls));tuning.append(dict(outer_fold=int(fold),family=family,config=json.dumps({'alpha':alpha}),validation_cross_entropy=mean));trials.append((mean,-alpha,alpha))
   ll,_,alpha=min(trials);best[family]=(ll,{'alpha':alpha})
  trials=[]
  for spec in boost_specs:
   ls=[]
   for ii,vv,FF,yy in inner:
    pp,_=boost_predict(FF['shallow_boosting'],D,yy,ii,vv,spec);ls.extend(loss(yy[vv],pp))
   mean=float(np.mean(ls));tuning.append(dict(outer_fold=int(fold),family='shallow_boosting',config=json.dumps(spec),validation_cross_entropy=mean));trials.append((mean,tuple(spec.values()),spec))
  ll,_,spec=min(trials);best['shallow_boosting']=(ll,spec)
  chosen=min(FAMILIES,key=lambda m:(best[m][0],FAMILIES.index(m)))
  choices.append(dict(fold=int(fold),chosen_family=chosen,chosen_config=json.dumps(best[chosen][1]),inner_cross_entropy=best[chosen][0]))
  for family,(ll,spec) in best.items():
   if family=='shallow_boosting':pp,par=boost_predict(outerF[family],D,y,tr,te,spec)
   else:pp,par=linear_predict(outerF[family],D,y,tr,te,spec['alpha'])
   for label in [family]+(['nested_selector'] if family==chosen else []):
    q=pairs.iloc[te].rename(columns={'drug':'regimen','compound':'drug'}).copy();q['observed']=y[te];q['predicted']=pp;q['fold']=fold;q['model']=label;q['n_high']=np.rint(y[te]*q.treated_n_cells).astype(int);pred.append(q)
   pars.append(dict(fold=int(fold),family=family,**par))
  print('FOLD',fold,'selector',chosen,'config',best[chosen][1],flush=True)
 predictions=pd.concat(pred,ignore_index=True)
 predictions.to_csv(OUT/'outer_predictions.csv',index=False,float_format='%.17g')
 perf=calculate(predictions);perf.to_csv(OUT/'performance.csv',index=False)
 pd.DataFrame(tuning).to_csv(OUT/'inner_tuning.csv',index=False);pd.DataFrame(choices).to_csv(OUT/'selection_by_fold.csv',index=False)
 (OUT/'fold_references.json').write_text(json.dumps(refs,indent=2)+'\n');(OUT/'fitted_parameters.json').write_text(json.dumps(pars,indent=2)+'\n')
 # The existing four-feature reference must reconstruct to numerical precision.
 published=pd.read_csv(INPUT/'nested_predictions.csv',float_precision='round_trip')
 published=published[(published.analysis=='composite_z2')&(published.cohort=='time_specific')&(published.split=='frozen_outer')&(published.model=='parsimonious_additive')]
 merged=predictions[predictions.model=='parsimonious_additive'].merge(published,on=['cell_line','regimen'],validate='one_to_one',suffixes=('_new','_old'))
 assert len(merged)==243
 maxerr=float(np.max(abs(merged.predicted_new-merged.predicted_old)));assert maxerr<1e-7,maxerr
 diagnostics(predictions);bootstrap(predictions)
 (OUT/'checks.json').write_text(json.dumps(dict(plan_sha256=hashlib.sha256(plan).hexdigest(),n_pairs=243,n_lines=84,reference_max_prediction_error=maxerr,elapsed_seconds=time.time()-started),indent=2)+'\n')
 print(perf[perf.drug=='all'][['model','r2','mae']].to_string(index=False),flush=True)
 print('FINISHED',time.time()-started,flush=True)

if __name__=='__main__':main()
