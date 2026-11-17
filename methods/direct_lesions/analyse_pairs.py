#!/usr/bin/env python3
"""Fixed DDR score versus independently measured physical lesion localization."""
from pathlib import Path
import os
import argparse,json
import numpy as np
import pandas as pd

def rankdata(x):
    return pd.Series(np.asarray(x)).rank(method='average').to_numpy()

ORDER=['Ctrl','0hr','2hr','6hr','24hr','48hr']
ENDPOINTS=['induced_fraction','conserved_fraction','induced_conserved_density_log2']

def rho(x,y,cov=None):
    x=rankdata(x);y=rankdata(y)
    if cov is not None:
        z=np.column_stack([np.ones(len(x))]+[rankdata(cov[:,i]) for i in range(cov.shape[1])])
        x=x-z@np.linalg.lstsq(z,x,rcond=None)[0];y=y-z@np.linalg.lstsq(z,y,rcond=None)[0]
    if np.std(x)<1e-10 or np.std(y)<1e-10:return np.nan
    return float(np.corrcoef(x,y)[0,1])

def analyse(d,out):
    out.mkdir(parents=True,exist_ok=True)
    facts=json.loads((Path(os.environ.get('DDR_LESION_ROOT',Path(__file__).parent))/'source_facts.json').read_text())
    ni=facts['physical_regions']['induced_union']['basepairs'];nc=facts['physical_regions']['conserved']['basepairs']
    assert (d.DNA_induced_union<=d.DNA_fragments_autosomal).all()
    assert (d.DNA_conserved<=d.DNA_fragments_autosomal).all()
    d['induced_fraction']=d.DNA_induced_union/d.DNA_fragments_autosomal
    d['conserved_fraction']=d.DNA_conserved/d.DNA_fragments_autosomal
    d['induced_conserved_density_log2']=np.log2(((d.DNA_induced_union+.5)/ni)/((d.DNA_conserved+.5)/nc))
    d['log_RNA_depth']=np.log1p(d.RNA_counts_deposited)
    d['log_RNA_genes']=np.log1p(d.RNA_genes_deposited)
    d['log_DNA_depth']=np.log1p(d.DNA_fragments_autosomal)
    basecov=['log_RNA_depth','log_RNA_genes','mitochondrial_fraction','log_DNA_depth']
    cyclecov=basecov+['cycle','E2F','S_Score','G2M_Score']
    results=[];capture=[];labels=[];summaries=[]
    for condition in ORDER:
        g=d[d.Treatment.eq(condition)].copy()
        if not len(g):continue
        g=g.replace([np.inf,-np.inf],np.nan).dropna(subset=['DDR34']+ENDPOINTS+cyclecov)
        design=pd.get_dummies(g[['capture_library','barcode_tube','Phase']].astype(str),drop_first=True,dtype=float).to_numpy()
        qc=np.column_stack([g[basecov].to_numpy(),design]);cy=np.column_stack([g[cyclecov].to_numpy(),design])
        for endpoint in ENDPOINTS:
            results.append({'condition':condition,'endpoint':endpoint,'n_cells':len(g),'n_DDR_high':int(g.DDR_high.sum()),'rho_unadjusted':rho(g.DDR34,g[endpoint]),'rho_QC':rho(g.DDR34,g[endpoint],qc),'rho_cycle_QC':rho(g.DDR34,g[endpoint],cy),'median_high':float(g.loc[g.DDR_high,endpoint].median()),'median_other':float(g.loc[~g.DDR_high,endpoint].median())})
            for library,a in g.groupby('capture_library'):
                zz=pd.get_dummies(a[['barcode_tube','Phase']].astype(str),drop_first=True,dtype=float).to_numpy()
                cov=np.column_stack([a[cyclecov].to_numpy(),zz])
                capture.append({'condition':condition,'capture_library':library,'endpoint':endpoint,'n_cells':len(a),'rho_cycle_QC':rho(a.DDR34,a[endpoint],cov)})
        h=g[g.DDR_high];lo=g[~g.DDR_high]
        labels.append({'condition':condition,'n_cells':len(g),'n_high':len(h),'high_fraction':len(h)/len(g),'induced_high_pseudobulk':h.DNA_induced_union.sum()/h.DNA_fragments_autosomal.sum() if len(h) else np.nan,'induced_other_pseudobulk':lo.DNA_induced_union.sum()/lo.DNA_fragments_autosomal.sum(),'conserved_high_pseudobulk':h.DNA_conserved.sum()/h.DNA_fragments_autosomal.sum() if len(h) else np.nan,'conserved_other_pseudobulk':lo.DNA_conserved.sum()/lo.DNA_fragments_autosomal.sum()})
        summaries.append({'condition':condition,'n_cells':len(g),'n_RNA_high':int(g.DDR_high.sum()),'mean_DDR34':float(g.DDR34.mean()),'median_induced_fraction':float(g.induced_fraction.median()),'pseudobulk_induced_fraction':float(g.DNA_induced_union.sum()/g.DNA_fragments_autosomal.sum()),'pseudobulk_conserved_fraction':float(g.DNA_conserved.sum()/g.DNA_fragments_autosomal.sum()),'DNA_fragments_total':int(g.DNA_fragments_autosomal.sum())})
    pd.DataFrame(results).to_csv(out/'condition_associations.csv',index=False,float_format='%.17g')
    pd.DataFrame(capture).to_csv(out/'capture_library_associations.csv',index=False,float_format='%.17g')
    pd.DataFrame(labels).to_csv(out/'fixed_control_label_contrasts.csv',index=False,float_format='%.17g')
    pd.DataFrame(summaries).to_csv(out/'condition_summaries.csv',index=False,float_format='%.17g')
    # Score quartiles are assigned within condition/tube before aggregating physical reads.
    d['DDR_quartile']=d.groupby(['Treatment','barcode_tube']).DDR34.transform(lambda x:pd.qcut(x.rank(method='first'),4,labels=False)+1)
    def reduce(g):
        return pd.Series({'n_cells':len(g),'mean_DDR34':g.DDR34.mean(),'DNA_auto':g.DNA_fragments_autosomal.sum(),'induced_reads':g.DNA_induced_union.sum(),'conserved_reads':g.DNA_conserved.sum()})
    agg=d.groupby(['Treatment','barcode_tube','capture_library','DDR_quartile']).apply(reduce,include_groups=False).reset_index()
    agg['induced_fraction']=agg.induced_reads/agg.DNA_auto;agg['conserved_fraction']=agg.conserved_reads/agg.DNA_auto
    agg.to_csv(out/'quartile_pseudobulk.csv',index=False,float_format='%.17g')
    d.to_csv(out/'paired_cells.csv.gz',index=False,float_format='%.17g')
    print(pd.DataFrame(results).to_string(index=False),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path(os.environ.get('DDR_LESION_ROOT',Path(__file__).parent))
    if a.input:d=pd.read_csv(a.input,low_memory=False,float_precision='round_trip')
    else:
        rna=pd.read_csv(root/'scored_rna.csv.gz',float_precision='round_trip');dna=pd.read_csv(root/'dna_counts.csv.gz',float_precision='round_trip')
        d=rna.merge(dna,on='barcode',validate='one_to_one')
        assert len(d)==64794
        print('Source cells with zero autosomal physical denominator',int(d.DNA_fragments_autosomal.eq(0).sum()),flush=True)
    analyse(d,a.output)

if __name__=='__main__':main()
