#!/usr/bin/env python3
from pathlib import Path
import os
import json,time,hashlib
import h5py,numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
import evaluate_extensions as ev
R=Path(os.environ['DDR_EXTENSION_OUTPUT']).resolve()
PHARM={'activation':['DCK','SLC29A1'],'inactivation':['CDA'],'ribonucleotide_reductase':['RRM1','RRM2']}

def control_scores(meta,controls,treated):
 ci=np.unique(np.concatenate(list(controls.values())))
 if not os.environ.get('DDR_EXTENSION_RAW'):
  cell=pd.read_csv(Path(os.environ['DDR_EXTENSION_RECORDS'])/'pharmacology_control_scores.csv.gz',float_precision='round_trip').set_index('source_row')
  assert set(cell.index)==set(ci) and cell.index.is_unique
  cell=cell.loc[ci];assert np.array_equal(cell.original_source_row.to_numpy(int),meta.iloc[ci].original_source_row.to_numpy(int))
  full=np.full((len(meta),len(PHARM)),np.nan);full[ci]=cell[list(PHARM)].to_numpy(float)
  assert np.isfinite(full[ci]).all()
  return full
 raw=Path(os.environ['DDR_EXTENSION_RAW'])
 with h5py.File(raw) as f:
  genes=ev.core.read_col(f['var'],'_index').astype(str);lookup={g:i for i,g in enumerate(genes)}
  wanted=sorted(set(sum(PHARM.values(),[])));assert all(g in lookup for g in wanted)
  x=f['X'];row_lookup=np.full(int(x.attrs['shape'][0]),-1,int);row_lookup[meta.iloc[ci].original_source_row.to_numpy(int)]=np.arange(len(ci));arr=np.zeros((len(ci),len(wanted)),float)
  for j,g in enumerate(wanted):
   a,b=map(int,x['indptr'][lookup[g]:lookup[g]+2]);rr=x['indices'][a:b];v=x['data'][a:b];which=row_lookup[rr];good=which>=0;arr[which[good],j]=v[good]
  arr=np.log1p(arr/meta.iloc[ci].ncounts.to_numpy(float)[:,None]*10000)
  out=np.column_stack([arr[:,[wanted.index(g) for g in gl]].mean(1) for gl in PHARM.values()])
  cell=pd.DataFrame(out,columns=list(PHARM));cell['source_row']=ci;cell['original_source_row']=meta.iloc[ci].original_source_row.to_numpy(int)
  cell.to_csv(R/'pharmacology_control_scores.csv.gz',index=False,float_format='%.17g')
  full=np.full((len(meta),len(PHARM)),np.nan);full[ci]=out
 return full

def features(F,pairs,controls,trainlines,pharm,Gall):
 ci=np.unique(np.concatenate([ix for (line,drug),ix in controls.items() if line in set(trainlines)]));a=pharm[ci];assert np.isfinite(a).all();cen=np.median(a,0);sc=1.4826*np.median(abs(a-cen),0);bad=sc<=0;cen[bad]=a.mean(0)[bad];sc[bad]=a.std(0)[bad];sc=np.maximum(sc,1e-8)
 z=(pharm-cen)/sc;B=np.vstack([z[controls[(line,drug)]].mean(0) for line,drug in zip(pairs.cell_line,pairs.drug)]);B*=pairs.compound.eq('Gemcitabine').to_numpy()[:,None]
 return {'CCLE_TP53_all_regimens':Gall,'state_CCLE_TP53_all_regimens':np.column_stack([F['parsimonious_additive'],Gall]),'state_CCLE_TP53_Gem_pharmacology':np.column_stack([F['parsimonious_additive'],Gall,B]),'CCLE_TP53_Gem_pharmacology_only':np.column_stack([Gall,B])},dict(n_training_controls=len(ci),pharmacology_centers=cen.tolist(),pharmacology_scales=sc.tolist(),module_names=list(PHARM))

def main():
 start=time.time();plan=(Path(os.environ['DDR_EXTENSION_RECORDS'])/'uniform_genotype_plan.json').read_bytes();print('FROZEN BIOLOGY PLAN',hashlib.sha256(plan).hexdigest(),flush=True)
 meta,names,modules,scores,pairs,controls,treated,folds,regimes,D,G=ev.load();pharm=control_scores(meta,controls,treated)
 source=pd.read_csv(Path(os.environ['DDR_EXTENSION_RECORDS'])/'uniform_TP53_annotations.csv').set_index('cell_line');known=pairs.cell_line.isin(source.index).to_numpy();nohot=np.array([float(source.loc[c,'TP53_protein_altering']==0) if c in source.index else 0 for c in pairs.cell_line]);unknown=(~known).astype(float)
 assert known.all() and source.TP53_profiled.all() and pairs.cell_line.nunique()==84
 Gall=np.column_stack([D*nohot[:,None],D*unknown[:,None]])
 pairs['TP53_protein_altering']=pairs.cell_line.map(source.TP53_protein_altering).astype(int)
 source.to_csv(R/'uniform_TP53_annotations.csv')
 coverage=pairs[['cell_line','drug','compound']].copy();coverage['known']=known;coverage['no_damaging_hotspot']=nohot;coverage['unknown']=unknown;coverage.to_csv(R/'uniform_TP53_coverage_by_observation.csv',index=False)
 coverage.groupby('drug').agg(n_observations=('known','size'),known_observations=('known','sum'),no_annotation_observations=('no_damaging_hotspot','sum'),unknown_observations=('unknown','sum'),n_lines=('cell_line','nunique')).to_csv(R/'uniform_TP53_coverage_by_regimen.csv')
 oldtune=pd.read_csv(R/'transcriptome_inner_tuning.csv');oldchoices=pd.read_csv(R/'transcriptome_selection_by_fold.csv');oldpred=pd.read_csv(R/'prior_family_predictions.csv',float_precision='round_trip')
 allpred=[];params=[];refs=[];tuning=[];choices=[]
 for fold in sorted(set(folds)):
  tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold);tls=set(pairs.iloc[tr].cell_line)
  F,y,ref=ev.features(scores,names,modules,pairs,controls,treated,tls,G);B,pr=features(F,pairs,controls,tls,pharm,Gall);refs.append(dict(level='outer',fold=int(fold),**ref,**pr))
  inner=[]
  for j,(itr,iva) in enumerate(GroupKFold(3).split(tr,groups=pairs.iloc[tr].cell_line),1):
   ii,vv=tr[itr],tr[iva];tls=set(pairs.iloc[ii].cell_line);FF,yy,rr=ev.features(scores,names,modules,pairs,controls,treated,tls,G);BB,pp=features(FF,pairs,controls,tls,pharm,Gall);inner.append((ii,vv,BB,yy));refs.append(dict(level='inner',fold=int(fold),inner_fold=j,**rr,**pp))
  trials={}
  for family in B:
   for alpha in [.3,3,30]:
    ce=[];mse=[]
    for ii,vv,BB,yy in inner:
     p,_=ev.linear_predict(BB[family],D,yy,ii,vv,alpha);ce.extend(ev.loss(yy[vv],p));mse.extend((yy[vv]-p)**2)
    e=dict(fold=int(fold),family=family,alpha=alpha,cross_entropy=float(np.mean(ce)),mse=float(np.mean(mse)));trials.setdefault(family,[]).append(e);tuning.append(e)
  for criterion in ['cross_entropy','mse']:
   best={f:min(tt,key=lambda e:(e[criterion],-e['alpha'])) for f,tt in trials.items()};ps={}
   for family,e in best.items():
    p,q=ev.linear_predict(B[family],D,y,tr,te,e['alpha']);ps[family]=p;params.append(dict(fold=int(fold),family=family,criterion=criterion,**q));a=pairs.iloc[te].copy().rename(columns={'drug':'regimen','compound':'drug'});a['observed']=y[te];a['predicted']=p;a['fold']=int(fold);a['model']=family+'__'+criterion;allpred.append(a)
   oldloss=float(oldtune[oldtune.fold.eq(fold)&oldtune.family.isin(ev.FAMILIES+['transcriptome_PC'])][criterion].min());newwinner=min(best,key=lambda f:best[f][criterion]);newloss=best[newwinner][criterion]
   if newloss<oldloss:
    winner=newwinner;pp=ps[winner];config=json.dumps({'alpha':best[winner]['alpha']})
   else:
    oc=oldchoices[(oldchoices.fold==fold)&(oldchoices.criterion==criterion)].iloc[0];winner=oc.augmented_family_winner;config=oc.selected_config
    z=oldpred[(oldpred.fold==fold)&oldpred.model.eq('selector_augmented_'+criterion)].set_index(['cell_line','regimen']);pp=np.array([z.loc[(pairs.iloc[i].cell_line,pairs.iloc[i].drug),'predicted'] for i in te])
   choices.append(dict(fold=int(fold),criterion=criterion,chosen_family=winner,chosen_config=config,inner_loss=min(oldloss,newloss)))
   a=pairs.iloc[te].copy().rename(columns={'drug':'regimen','compound':'drug'});a['observed']=y[te];a['predicted']=pp;a['fold']=int(fold);a['model']='selector_uniform_'+criterion;allpred.append(a)
  print('FOLD',fold,choices[-2:],flush=True)
 pd.DataFrame(tuning).to_csv(R/'uniform_biology_inner_tuning.csv',index=False);pd.DataFrame(choices).to_csv(R/'uniform_biology_selection_by_fold.csv',index=False)
 (R/'uniform_biology_fitted_parameters.json').write_text(json.dumps(params,indent=2)+'\n');(R/'uniform_biology_fold_references.json').write_text(json.dumps(refs,indent=2)+'\n')
 new=pd.concat(allpred,ignore_index=True);new.to_csv(R/'uniform_biology_outer_predictions.csv',index=False,float_format='%.17g');merged=pd.concat([oldpred,new],ignore_index=True);merged.to_csv(R/'uniform_all_outer_predictions.csv',index=False,float_format='%.17g');perf=ev.calculate(merged);perf.to_csv(R/'uniform_all_performance.csv',index=False)
 pd.DataFrame(tuning).to_csv(R/'uniform_biology_inner_tuning.csv',index=False);pd.DataFrame(choices).to_csv(R/'uniform_biology_selection_by_fold.csv',index=False)
 (R/'uniform_biology_checks.json').write_text(json.dumps(dict(plan_sha256=hashlib.sha256(plan).hexdigest(),n_pairs=len(pairs),n_lines=pairs.cell_line.nunique(),TP53_known_lines=int(pairs[pairs.cell_line.isin(source.index)].cell_line.nunique()),TP53_unknown_lines=int(pairs[~pairs.cell_line.isin(source.index)].cell_line.nunique()),elapsed_seconds=time.time()-start),indent=2)+'\n')
 print(perf[(perf.drug=='all')&perf.model.str.contains('TP53_|selector_final')][['model','r2','mae']].to_string(index=False),flush=True)

if __name__=='__main__':main()
