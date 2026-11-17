#!/usr/bin/env python3
"""Prespecified physical-endpoint coverage and fixed-component checks.

Monte Carlo ranges quantify read-sampling variability, not biological uncertainty.
"""
from pathlib import Path
import os
import argparse
import numpy as np
import pandas as pd
from analyse_pairs import ORDER, ENDPOINTS, rho, rankdata

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=50);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.input,low_memory=False,float_precision='round_trip');rng=np.random.default_rng(20261117)
    continuous=['log_RNA_depth','log_RNA_genes','mitochondrial_fraction','log_DNA_depth','cycle','E2F','S_Score','G2M_Score']
    components=[];rarefied=[]
    for condition in ORDER:
        g=d[d.Treatment.eq(condition)].replace([np.inf,-np.inf],np.nan).dropna(subset=['DDR34','p53','repair']+ENDPOINTS+continuous).copy()
        design=pd.get_dummies(g[['capture_library','barcode_tube','Phase']].astype(str),drop_first=True,dtype=float).to_numpy()
        cov=np.column_stack([g[continuous].to_numpy(),design])
        for score in ['p53','repair']:
            for endpoint in ENDPOINTS:components.append({'condition':condition,'score':score,'endpoint':endpoint,'n_cells':len(g),'rho_unadjusted':rho(g[score],g[endpoint]),'rho_cycle_QC':rho(g[score],g[endpoint],cov)})
        for depth in [500,1000]:
            h=g[g.DNA_fragments_autosomal.ge(depth)].copy()
            design=pd.get_dummies(h[['capture_library','barcode_tube','Phase']].astype(str),drop_first=True,dtype=float).to_numpy()
            cov=np.column_stack([h[continuous].to_numpy(),design])
            z=np.column_stack([np.ones(len(h))]+[rankdata(cov[:,i]) for i in range(cov.shape[1])]);u,s,_=np.linalg.svd(z,full_matrices=False);q=u[:,s>s[0]*1e-12]
            x=rankdata(h.DDR34);x=x-q@(q.T@x)
            hits=h.DNA_induced_union.to_numpy(np.int64);total=h.DNA_fragments_autosomal.to_numpy(np.int64)
            for repeat in range(a.repeats):
                sampled=rng.hypergeometric(hits,total-hits,depth);y=rankdata(sampled);y=y-q@(q.T@y)
                rarefied.append({'condition':condition,'reads_per_cell':depth,'repeat':repeat,'n_cells':len(h),'rho_unadjusted':rho(h.DDR34,sampled),'rho_cycle_QC':float(np.corrcoef(x,y)[0,1])})
    pd.DataFrame(components).to_csv(a.output/'fixed_component_associations.csv',index=False,float_format='%.17g')
    r=pd.DataFrame(rarefied);r.to_csv(a.output/'equal_cell_depth_sensitivity.csv',index=False,float_format='%.17g')
    r.groupby(['condition','reads_per_cell']).agg(n_cells=('n_cells','first'),rho_median=('rho_cycle_QC','median'),rho_min=('rho_cycle_QC','min'),rho_max=('rho_cycle_QC','max')).reset_index().to_csv(a.output/'equal_cell_depth_summary.csv',index=False,float_format='%.17g')
    # Four score quartiles are read-matched within condition, initial tube and capture library.
    q=pd.read_csv(a.output/'quartile_pseudobulk.csv',float_precision='round_trip');q=q.sort_values(['Treatment','barcode_tube','capture_library','DDR_quartile']).reset_index(drop=True)
    grouping=['Treatment','barcode_tube','capture_library'];q['matched_reads']=q.groupby(grouping).DNA_auto.transform('min').astype(np.int64)
    assert q.groupby(grouping).DDR_quartile.nunique().eq(4).all()
    draws=[]
    for repeat in range(a.repeats):
        for endpoint in ['induced','conserved']:
            hit=q[endpoint+'_reads'].to_numpy(np.int64);total=q.DNA_auto.to_numpy(np.int64);depth=q.matched_reads.to_numpy(np.int64)
            k=q[grouping+['DDR_quartile','matched_reads']].copy();k['matched_hits']=rng.hypergeometric(hit,total-hit,depth)
            for label,g in k.groupby('Treatment'):
                agg=g.groupby('DDR_quartile')[['matched_hits','matched_reads']].sum();fraction=agg.matched_hits/agg.matched_reads
                draws.append({'condition':label,'endpoint':endpoint,'repeat':repeat,'Q1_fraction':fraction.loc[1],'Q4_fraction':fraction.loc[4],'Q4_Q1_ratio':fraction.loc[4]/fraction.loc[1]})
    rr=pd.DataFrame(draws);rr.to_csv(a.output/'read_matched_pseudobulk.csv',index=False,float_format='%.17g')
    rr.groupby(['condition','endpoint']).agg(ratio_median=('Q4_Q1_ratio','median'),sampling_min=('Q4_Q1_ratio','min'),sampling_max=('Q4_Q1_ratio','max')).reset_index().to_csv(a.output/'read_matched_summary.csv',index=False,float_format='%.17g')
    q.to_csv(a.output/'quartile_read_depths.csv',index=False,float_format='%.17g')
    print('Completed fixed component, equal-cell-depth and pseudobulk read-matching sensitivity checks.',flush=True)
if __name__=='__main__':main()
