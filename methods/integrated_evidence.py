#!/usr/bin/env python3
"""Recompute the fixed model extension and paired physical/function endpoints."""
from pathlib import Path
import json,os,subprocess,sys
import numpy as np,pandas as pd

def compare_csv(reference,actual):
 pd.testing.assert_frame_equal(pd.read_csv(reference,float_precision='round_trip'),pd.read_csv(actual,float_precision='round_trip'),check_dtype=False,check_exact=False,atol=1e-12,rtol=1e-12)

def run_integrated(inputs,results,methods,args):
 def run(script,*values,env=None):subprocess.run([sys.executable,str(methods/script),*map(str,values)],check=True,env=env)
 ext=results/'model_extension';options=[]
 if args.refit=='extension' or args.raw=='extension':options.append('--refit')
 if args.raw=='extension':options.extend(['--raw',args.data_root/'datasets/scperturb/McFarlandTsherniak2020.h5ad'])
 run('model_extension/reproduce_extension.py','--forecast-inputs',inputs/'forecast','--context-inputs',inputs/'source_context','--methods',methods,'--records',inputs/'model_extension','--output',ext,*options)
 run('model_extension/rebuild_genotype_annotations.py','--records',inputs/'model_extension','--output',ext)
 p=pd.read_csv(ext/'curated_all_outer_predictions.csv',float_precision='round_trip');main=p[p.model.eq('state_curated_TP53_Gem_pharmacology__cross_entropy')]
 assert len(main)==243 and main.cell_line.nunique()==84 and not main.duplicated(['cell_line','regimen']).any() and main.groupby('cell_line').fold.nunique().max()==1
 assert main.predicted.between(0,1).all()
 import nested_forecasting as core
 m=core.metrics(main.observed.to_numpy(),main.predicted.to_numpy());assert abs(m['r2']-.6231631021548982)<1e-7
 gene_args=['--counts',args.data_root/'datasets/direct_damage/GSE250285_counts_rpe_dox_gp.tsv.gz'] if args.raw=='damage' else ['--gene-input',inputs/'direct_damage/H2AFX_scored_cells.csv.gz']
 run('h2afx_robustness.py','--cells',inputs/'direct_damage/all_scored_cells.csv.gz',*gene_args,'--output',results/'direct_damage')
 compare_csv(inputs/'direct_damage/H2AFX_shared_gene_sensitivity.csv',results/'direct_damage/H2AFX_shared_gene_sensitivity.csv')
 env=os.environ.copy();env['DDR_LESION_ROOT']=str((inputs/'direct_lesions').resolve())
 if args.raw!='lesions':
  for reference in (inputs/'direct_lesions').glob('*.csv'):
   if reference.name!='gene_coverage.csv':(results/'direct_lesions'/reference.name).unlink(missing_ok=True)
  run('direct_lesions/analyse_pairs.py','--input',inputs/'direct_lesions/paired_cells.csv.gz','--output',results/'direct_lesions',env=env)
 run('direct_lesions/sensitivity.py','--input',results/'direct_lesions/paired_cells.csv.gz','--output',results/'direct_lesions',env=env)
 checks=[]
 for reference in (inputs/'direct_lesions').glob('*.csv'):
  actual=results/'direct_lesions'/reference.name
  if actual.exists() and reference.name not in ['gene_coverage.csv']:
   compare_csv(reference,actual);checks.append(reference.name)
 assert len(checks)==11
 return m
