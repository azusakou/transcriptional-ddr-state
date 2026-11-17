#!/usr/bin/env python3
"""Shared-gene sensitivity for the same-cell RNA–gammaH2AX association.

Use the deposited H2AFX transcript count row, or its compact extracted table,
alongside the complete scored-cell input. The original 33-gene composite and
its primary analyses are retained unchanged.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def partial_rank(x, y, covariates):
    a,b=rankdata(x),rankdata(y)
    z=np.column_stack([np.ones(len(a))]+[rankdata(v) for v in covariates])
    a=a-z@np.linalg.lstsq(z,a,rcond=None)[0]
    b=b-z@np.linalg.lstsq(z,b,rcond=None)[0]
    return float(np.corrcoef(a,b)[0,1])


def extract_count_row(path):
    with gzip.open(path,'rt') as stream:
        header=stream.readline().rstrip('\r\n').split('\t')
        if header[0]=='':header=header[1:]
        found=[]
        for line in stream:
            name,_,values=line.partition('\t')
            if name in ['H2AFX','H2AX']:
                counts=np.fromstring(values,sep='\t')
                assert len(counts)==len(header)
                found.append((name,counts))
    assert len(found)==1,'Expected one source column identity for H2AFX/H2AX'
    return pd.DataFrame({'cell_id':header,'H2AFX_transcript_count':found[0][1]})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells',type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--counts',type=Path,help='Deposited complete count TSV.gz')
    group.add_argument('--gene-input',type=Path,help='Compact extracted H2AFX table')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cells=pd.read_csv(args.cells,float_precision='round_trip')
    gene=extract_count_row(args.counts) if args.counts else pd.read_csv(args.gene_input,float_precision='round_trip')
    assert cells.cell_id.is_unique and gene.cell_id.is_unique
    assert set(cells.cell_id)==set(gene.cell_id)
    df=cells.merge(gene[['cell_id','H2AFX_transcript_count']],on='cell_id',validate='one_to_one')
    df['H2AFX_log1p10k']=np.log1p(10000*df.H2AFX_transcript_count/df.endogenous_counts)
    # GADD45G is not represented in this source; H2AFX is one of 33 observed
    # fixed DDR genes. Leave-one-gene-out membership is fixed by the assay.
    df['DDR_without_H2AFX']=(33*df.DDR-df.H2AFX_log1p10k)/32
    assert np.all(df.DDR_without_H2AFX>=-1e-12)
    aux=df[['cell_id','H2AFX_transcript_count','H2AFX_log1p10k','DDR_without_H2AFX']]
    aux.to_csv(args.output/'H2AFX_scored_cells.csv.gz',index=False,float_format='%.17g')
    used=df[df.qc_pass.astype(bool)&df.treatment.eq('GP')].copy()
    assert len(used)==1514 and used.experimentid.nunique()==6
    rows=[]
    for plate,p in used.groupby('experimentid',sort=True):
        cov=[p.DAPI_fluorescence,p.log_depth,p.log_genes]
        rows.append(dict(
            plate=plate,hras=p.hras.iloc[0],n_cells=len(p),
            primary_partial_rho=partial_rank(p.DDR,p.yH2AX_fluorescence,cov),
            primary_cycle_partial_rho=partial_rank(p.DDR,p.yH2AX_fluorescence,cov+[p.cycle]),
            without_H2AFX_partial_rho=partial_rank(p.DDR_without_H2AFX,p.yH2AX_fluorescence,cov),
            without_H2AFX_cycle_partial_rho=partial_rank(p.DDR_without_H2AFX,p.yH2AX_fluorescence,cov+[p.cycle]),
            H2AFX_RNA_adjusted_partial_rho=partial_rank(p.DDR,p.yH2AX_fluorescence,cov+[p.H2AFX_log1p10k]),
            H2AFX_RNA_cycle_adjusted_partial_rho=partial_rank(p.DDR,p.yH2AX_fluorescence,cov+[p.cycle,p.H2AFX_log1p10k]),
            H2AFX_RNA_only_partial_rho=partial_rank(p.H2AFX_log1p10k,p.yH2AX_fluorescence,cov),
        ))
    result=pd.DataFrame(rows)
    result.to_csv(args.output/'H2AFX_shared_gene_sensitivity.csv',index=False,float_format='%.17g')
    facts={
        'assay':'GSE250285 same-cell RNA and gammaH2AX fluorescence',
        'membership':'Frozen 33 observed DDR markers; remove the one transcript encoding the measured phosphoprotein substrate, H2AFX (approved symbol H2AX), retaining 32 markers.',
        'normalization':'log1p(10000 * H2AFX source transcript count / complete endogenous transcript total)',
        'adjustment':'Within-plate partial rank correlation adjusted for ranked DAPI, log endogenous depth and log detected genes; cycle score and H2AFX RNA abundance added in the specified sensitivities.',
        'unit':'Selected cells within six sorting plates; no biological-culture replication inferred.',
        'interpretation':'RNA abundance adjustment does not directly measure or control total H2AX protein abundance.',
    }
    if args.counts:
        facts['source_count_filename']=args.counts.name
        facts['source_count_sha256']=hashlib.sha256(args.counts.read_bytes()).hexdigest()
    (args.output/'H2AFX_shared_gene_definitions.json').write_text(json.dumps(facts,indent=2)+'\n')
    print(result.to_string(index=False))


if __name__=='__main__':main()
