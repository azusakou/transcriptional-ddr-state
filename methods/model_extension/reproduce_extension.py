#!/usr/bin/env python3
"""Reproduce fixed biological extensions, with optional exact nested refitting."""
from pathlib import Path
import argparse,gzip,json,os,runpy,shutil,sys

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['forecast-inputs','context-inputs','methods','records','output']:
  p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--refit',action='store_true',help='Refit final four biological families using compact source-derived inputs')
 p.add_argument('--raw',type=Path,help='Optional original McFarland H5AD: rederive the five pharmacology genes instead of reading compact scores')
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 for name,path in [('FORECAST_INPUTS',a.forecast_inputs),('CONTEXT_INPUTS',a.context_inputs),('METHODS',a.methods),('RECORDS',a.records),('OUTPUT',a.output)]:os.environ['DDR_EXTENSION_'+name]=str(path.resolve())
 if a.raw:os.environ['DDR_EXTENSION_RAW']=str(a.raw.resolve())
 else:os.environ.pop('DDR_EXTENSION_RAW',None)
 code=Path(__file__).resolve().parent;sys.path[:0]=[str(code),str(a.methods.resolve())]
 import numpy as np,pandas as pd
 import evaluate_extensions as ev
 from scipy.special import expit
 def unpack(name):
  with gzip.open(a.records/(name+'.gz'),'rb') as source,open(a.output/name,'wb') as dest:shutil.copyfileobj(source,dest)
 for name in ['transcriptome_inner_tuning.csv','transcriptome_selection_by_fold.csv']:
  shutil.copy2(a.records/name,a.output/name)
 unpack('prior_family_predictions.csv')
 for prefix in ['curated','uniform']:
  if a.refit:runpy.run_path(str(code/('evaluate_'+prefix+'_biology.py')),run_name='__main__')
  else:
   unpack(prefix+'_all_outer_predictions.csv')
   for suffix in ['fitted_parameters','fold_references']:unpack(prefix+'_biology_'+suffix+'.json')
   shutil.copy2(a.records/(prefix+'_TP53_annotations.csv'),a.output/(prefix+'_TP53_annotations.csv'))
  pred=pd.read_csv(a.output/(prefix+'_all_outer_predictions.csv'),float_precision='round_trip')
  ev.calculate(pred).to_csv(a.output/(prefix+'_all_performance.csv'),index=False)
  runpy.run_path(str(code/(prefix+'_biology_intervals.py')),run_name='__main__')
 # Check all 40 fold/family/criterion parameter records for each annotation.
 meta,names,modules,scores,pairs,controls,treated,folds,regimes,D,G=ev.load();checks={}
 for prefix in ['curated','uniform']:
  bm=__import__('evaluate_'+prefix+'_biology');pharm=bm.control_scores(meta,controls,treated)
  ann=pd.read_csv(a.records/(prefix+'_TP53_annotations.csv')).set_index('cell_line')
  field='TP53_curated_qualifying' if prefix=='curated' else 'TP53_protein_altering'
  no_call=pairs.cell_line.map(ann[field]).eq(0).to_numpy(float)
  assert pairs.cell_line.isin(ann.index).all() and ann.TP53_profiled.all()
  Gall=np.column_stack([D*no_call[:,None],np.zeros_like(D)])
  params=json.loads((a.output/(prefix+'_biology_fitted_parameters.json')).read_text())
  pred=pd.read_csv(a.output/(prefix+'_all_outer_predictions.csv'),float_precision='round_trip');errors=[]
  for fold in sorted(set(folds)):
   tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold);tls=set(pairs.iloc[tr].cell_line)
   F,y,_=ev.features(scores,names,modules,pairs,controls,treated,tls,G);B,_=bm.features(F,pairs,controls,tls,pharm,Gall)
   for q in [q for q in params if q['fold']==fold]:
    X=np.column_stack([D,(B[q['family']]-q['feature_center'])/q['feature_scale']]);pp=expit(X[te]@np.array(q['coefficients']))
    z=pred[pred.fold.eq(fold)&pred.model.eq(q['family']+'__'+q['criterion'])].set_index(['cell_line','regimen'])
    target=np.array([z.loc[(pairs.iloc[i].cell_line,pairs.iloc[i].drug),'predicted'] for i in te]);errors.append(float(np.max(abs(pp-target))))
    assert errors[-1]<1e-10
  assert len(params)==40
  checks[prefix]={'parameter_records':len(params),'OOF_max_reconstruction_error':max(errors),'n_observations':len(pairs),'n_lines':pairs.cell_line.nunique(),'qualifying_lines':int(ann[field].sum()),'no_qualifying_call_lines':int((ann[field]==0).sum()),'unknown_lines':0}
  if a.refit:
   frozen=pd.read_csv(a.records/(prefix+'_all_outer_predictions.csv.gz'),float_precision='round_trip').set_index(['model','cell_line','regimen']).sort_index()
   new=pred.set_index(['model','cell_line','regimen']).sort_index();error=float(np.max(abs(new.predicted-frozen.predicted)));assert error<1e-10
   checks[prefix]['fresh_nested_refit_max_OOF_error']=error
 (a.output/'reproduction_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
 runpy.run_path(str(code/'primary_model_diagnostics.py'),run_name='__main__')
 print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
