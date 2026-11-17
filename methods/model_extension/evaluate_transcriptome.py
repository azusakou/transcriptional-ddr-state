#!/usr/bin/env python3
from pathlib import Path
import os
import json,time,itertools,hashlib
import numpy as np,pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from scipy.special import expit
import evaluate_extensions as ev

R=Path(os.environ['DDR_EXTENSION_OUTPUT']).resolve()
T=np.load(R/'control_transcriptome.npz');E=T['means'];GENES=T['genes'];PG=T['pair_groups']

def pc_features(pairs,trainlines):
 train=np.flatnonzero(pairs.cell_line.isin(set(trainlines)));ug=np.unique(PG[train]);x=E[ug]
 good=np.mean(x>0,axis=0)>=.1;var=np.var(x,axis=0);order=np.argsort(-var,kind='stable');idx=order[good[order]][:2000]
 assert len(idx)==2000
 cen=x[:,idx].mean(0);sc=x[:,idx].std(0);sc[sc<1e-8]=1
 model=PCA(n_components=6,svd_solver='full');model.fit((x[:,idx]-cen)/sc)
 pp=model.transform((E[PG][:,idx]-cen)/sc)
 record=dict(n_training_control_populations=len(ug),n_variance_selected_genes=len(idx),source_gene_indices=idx.tolist(),gene_symbols=GENES[idx].tolist(),gene_means=cen.tolist(),gene_scales=sc.tolist(),pca_mean=model.mean_.tolist(),pca_components=model.components_.tolist(),pca_explained_variance_ratio=model.explained_variance_ratio_.tolist())
 return pp,record

def pc_predict(B,PC,D,y,tr,te,k,alpha):
 bb=B;xx=np.column_stack([bb,PC[:,:k]])
 cen=xx[tr].mean(0);sc=xx[tr].std(0);sc[sc<1e-12]=1;z=(xx-cen)/sc
 interaction=((D-1/D.shape[1])[:,:,None]*z[:,None,4:]).reshape(len(B),-1)
 X=np.column_stack([D,z,interaction]);pen=np.r_[np.zeros(D.shape[1]),np.ones(z.shape[1]),4*np.ones(interaction.shape[1])]
 beta,ok,nit=ev.core.fit_model(X[tr],y[tr],pen,alpha)
 return expit(X[te]@beta),dict(k=k,alpha=alpha,feature_center=cen.tolist(),feature_scale=sc.tolist(),coefficients=beta.tolist(),converged=bool(ok))

def main():
 started=time.time();plan=(R/'transcriptome_plan.json').read_bytes();print('PCA PLAN',hashlib.sha256(plan).hexdigest(),flush=True)
 meta,names,modules,scores,pairs,controls,treated,folds,regimes,D,G=ev.load()
 boost_specs=[dict(max_leaf_nodes=l,min_samples_leaf=n,l2_regularization=a) for l,n,a in itertools.product([3,5],[8,16],[3,30])]
 pred=[];tuning=[];choices=[];records=[]
 for fold in sorted(set(folds)):
  tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold);trainlines=set(pairs.iloc[tr].cell_line)
  F,y,ref=ev.features(scores,names,modules,pairs,controls,treated,trainlines,G);PC,pcfit=pc_features(pairs,trainlines)
  inner=[]
  for j,(itr,iva) in enumerate(GroupKFold(3).split(tr,groups=pairs.iloc[tr].cell_line),1):
   ii,vv=tr[itr],tr[iva];tls=set(pairs.iloc[ii].cell_line);FF,yy,rr=ev.features(scores,names,modules,pairs,controls,treated,tls,G);pp,qq=pc_features(pairs,tls);inner.append((ii,vv,FF,yy,pp))
  trials={}
  for fam,alphas in ev.ALPHAS.items():
   for alpha in alphas:
    ce=[];mse=[]
    for ii,vv,FF,yy,pp in inner:
     p,_=ev.linear_predict(FF[fam],D,yy,ii,vv,alpha);ce.extend(ev.loss(yy[vv],p));mse.extend((yy[vv]-p)**2)
    trials.setdefault(fam,[]).append(dict(config={'alpha':alpha},cross_entropy=float(np.mean(ce)),mse=float(np.mean(mse))))
  for spec in boost_specs:
   ce=[];mse=[]
   for ii,vv,FF,yy,pp in inner:
    p,_=ev.boost_predict(FF['shallow_boosting'],D,yy,ii,vv,spec);ce.extend(ev.loss(yy[vv],p));mse.extend((yy[vv]-p)**2)
   trials.setdefault('shallow_boosting',[]).append(dict(config=spec,cross_entropy=float(np.mean(ce)),mse=float(np.mean(mse))))
  for k,alpha in itertools.product([3,6],[.3,3,30]):
   ce=[];mse=[]
   for ii,vv,FF,yy,pp in inner:
    p,_=pc_predict(FF['parsimonious_additive'],pp,D,yy,ii,vv,k,alpha);ce.extend(ev.loss(yy[vv],p));mse.extend((yy[vv]-p)**2)
   trials.setdefault('transcriptome_PC',[]).append(dict(config={'k':k,'alpha':alpha},cross_entropy=float(np.mean(ce)),mse=float(np.mean(mse))))
  family_order=ev.FAMILIES+['transcriptome_PC']
  outer={};par={}
  for criterion in ['cross_entropy','mse']:
   selected={fam:min(tt,key=lambda z:(z[criterion],len(str(z['config'])))) for fam,tt in trials.items()}
   oldwinner=min(ev.FAMILIES,key=lambda fam:(selected[fam][criterion],ev.FAMILIES.index(fam)))
   winner=min(family_order,key=lambda fam:(selected[fam][criterion],family_order.index(fam)))
   choices.append(dict(fold=int(fold),criterion=criterion,original_family_winner=oldwinner,augmented_family_winner=winner,selected_config=json.dumps(selected[winner]['config']),validation_loss=selected[winner][criterion]))
   for fam,entry in selected.items():
    spec=entry['config'];tag=fam+'__'+criterion
    if fam=='shallow_boosting':p,q=ev.boost_predict(F[fam],D,y,tr,te,spec)
    elif fam=='transcriptome_PC':p,q=pc_predict(F['parsimonious_additive'],PC,D,y,tr,te,**spec)
    else:p,q=ev.linear_predict(F[fam],D,y,tr,te,spec['alpha'])
    outer[tag]=p;par[tag]=q
   for label,fam in [('selector_original_'+criterion,oldwinner),('selector_augmented_'+criterion,winner),('transcriptome_PC_'+criterion,'transcriptome_PC')]:
    q=pairs.iloc[te].copy().rename(columns={'drug':'regimen','compound':'drug'});q['observed']=y[te];q['predicted']=outer[fam+'__'+criterion];q['fold']=int(fold);q['model']=label;pred.append(q)
   records.append(dict(fold=int(fold),criterion=criterion,winner=winner,family_parameters=par.copy(),outer_score_reference=ref,outer_pc_fit=pcfit))
  for fam,tt in trials.items():
   for e in tt:tuning.append(dict(fold=int(fold),family=fam,config=json.dumps(e['config']),cross_entropy=e['cross_entropy'],mse=e['mse']))
  print('FOLD',fold,choices[-2:],flush=True)
 added=pd.concat(pred,ignore_index=True);added.to_csv(R/'transcriptome_outer_predictions.csv',index=False,float_format='%.17g')
 old=pd.read_csv(R/'outer_predictions.csv',float_precision='round_trip');merged=pd.concat([old,added],ignore_index=True);merged.to_csv(R/'all_outer_predictions.csv',index=False,float_format='%.17g')
 perf=ev.calculate(merged);perf.to_csv(R/'all_performance.csv',index=False)
 pd.DataFrame(choices).to_csv(R/'transcriptome_selection_by_fold.csv',index=False);pd.DataFrame(tuning).to_csv(R/'transcriptome_inner_tuning.csv',index=False)
 (R/'transcriptome_fitted_parameters.json').write_text(json.dumps(records,indent=2)+'\n')
 reference=added[added.model.eq('selector_original_cross_entropy')].merge(old[old.model.eq('nested_selector')],on=['cell_line','regimen'],suffixes=('_new','_old'),validate='one_to_one')
 maxerr=float(np.max(abs(reference.predicted_new-reference.predicted_old)));assert maxerr<1e-7,maxerr
 # Emit confidence intervals for the added models against all original comparators.
 evalrows=merged.copy();evalrows.to_csv(R/'all_outer_predictions.csv',index=False,float_format='%.17g')
 (R/'transcriptome_checks.json').write_text(json.dumps(dict(reference_selector_max_error=maxerr,plan_sha256=hashlib.sha256(plan).hexdigest(),elapsed_seconds=time.time()-started),indent=2)+'\n')
 print(perf[perf.drug.eq('all')][['model','r2','mae']].to_string(index=False),flush=True);print('DONE',time.time()-started,flush=True)

if __name__=='__main__':main()
