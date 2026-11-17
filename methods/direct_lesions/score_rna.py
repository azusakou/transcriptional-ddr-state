#!/usr/bin/env python3
"""Score fixed expression constructs from deposited Paired-Damage RNA counts."""
from pathlib import Path
import os
import csv,gzip,hashlib,json,tarfile
import numpy as np
import pandas as pd

ROOT=Path(os.environ.get('DDR_LESION_ROOT',Path(__file__).parent))
extract=ROOT/'rna_matrix';extract.mkdir(exist_ok=True)
if not (extract/'matrix.mtx').exists():
    with tarfile.open(ROOT/'rna.gz','r:gz') as archive:
        for member in archive:
            name=Path(member.name).name
            if member.isfile() and name in ['matrix.mtx','genes.tsv','barcodes.tsv','features.tsv']:
                stream=archive.extractfile(member)
                with (extract/name).open('wb') as out:
                    for chunk in iter(lambda:stream.read(2**20),b''):out.write(chunk)
                print('Extracted',name,member.size,flush=True)

gene_path=extract/'genes.tsv'
if not gene_path.exists():gene_path=extract/'features.tsv'
genes=pd.read_csv(gene_path,sep='\t',header=None)
symbols=genes.iloc[:,1 if genes.shape[1]>1 else 0].astype(str).to_numpy()
bar=pd.read_csv(extract/'barcodes.tsv',sep='\t',header=None).iloc[:,0].astype(str).to_numpy()
assert len(bar)==64794 and len(set(bar))==len(bar)
meta=pd.read_csv(ROOT/'source_cell_metadata.csv').set_index('Barcodes')
assert set(bar)==set(meta.index)
meta=meta.loc[bar].copy()
target='TP53 CDKN1A MDM2 GADD45A GADD45B GADD45G DDB2 XPC BBC3 BAX PMAIP1 FAS RRM2B SESN1 BTG2 FDXR PLK3 PPM1D ATM ATR CHEK1 CHEK2 BRCA1 BRCA2 RAD51 PARP1 XRCC5 XRCC6 H2AFX MRE11 NBN FANCD2 CLSPN TOPBP1'.split()
p53='TP53 CDKN1A MDM2 DDB2 GADD45A GADD45B RRM2B BBC3 BAX PMAIP1 PPM1D FDXR SESN1 BTG2 FAS'.split()
cycle='MKI67 TOP2A PCNA MCM2 MCM3 MCM4 MCM5 MCM6 MCM7 CDK1 CCNB1 CCNB2 CCNE1 CCNA2 CDC20 AURKA AURKB'.split()
e2f='E2F1 E2F2 E2F3 MCM2 MCM3 MCM4 MCM5 MCM6 MCM7 PCNA RRM1 RRM2 TYMS TK1 CDC6 CCNE1 CCNA2 CDK2'.split()
sets={'DDR34':target,'p53':p53,'repair':sorted(set(target)-set(p53)),'cycle':cycle,'E2F':e2f}
identity=json.loads((ROOT/'gene_identity_records.json').read_text())
lookup={};coverage=[]
for gene in sorted(set(sum(sets.values(),[]))):
    direct=np.flatnonzero(symbols==gene)
    valid=identity['records'][gene]['valid_source_symbols']
    matches=np.flatnonzero(np.isin(symbols,valid))
    if len(direct)==1:index=int(direct[0])
    elif len(matches)==1:index=int(matches[0])
    elif len(matches)==0:index=None
    else:raise ValueError((gene,symbols[matches]))
    lookup[gene]=index
    coverage.append({'specified_gene':gene,'source_symbol':symbols[index] if index is not None else '', 'source_row':index,'HGNC_ID':identity['records'][gene]['hgnc_id']})
pd.DataFrame(coverage).to_csv(ROOT/'gene_coverage.csv',index=False)
chosen=sorted({x for x in lookup.values() if x is not None})
row_to_slot=np.full(len(symbols),-1,np.int32);row_to_slot[chosen]=np.arange(len(chosen))
marker=np.zeros((len(bar),len(chosen)),np.float64)
total=np.zeros(len(bar));ngene=np.zeros(len(bar));mt=np.zeros(len(bar))
mtflag=np.char.startswith(symbols.astype(str),'MT-')
matrix=extract/'matrix.mtx'
with matrix.open() as source:
    first=source.readline();assert 'coordinate integer general' in first
    shape=source.readline()
    while shape.startswith('%'):shape=source.readline()
    nrow,ncol,nnz=map(int,shape.split());assert nrow==len(symbols) and ncol==len(bar)
    for i,chunk in enumerate(pd.read_csv(source,sep=' ',header=None,dtype=np.int32,chunksize=2000000)):
        values=chunk.to_numpy();g=values[:,0]-1;c=values[:,1]-1;v=values[:,2]
        assert (v>0).all()
        total+=np.bincount(c,weights=v,minlength=len(bar));ngene+=np.bincount(c,minlength=len(bar))
        mask=mtflag[g];mt+=np.bincount(c[mask],weights=v[mask],minlength=len(bar))
        slot=row_to_slot[g];mask=slot>=0;marker[c[mask],slot[mask]]=v[mask]
        if i%10==0:print('Matrix rows processed',(i+1)*2000000,flush=True)

norm=np.log1p(marker*10000/np.maximum(total[:,None],1))
for name,members in sets.items():
    idx=[row_to_slot[lookup[g]] for g in members if lookup[g] is not None]
    meta[name]=norm[:,idx].mean(axis=1)
meta['RNA_counts_deposited']=total;meta['RNA_genes_deposited']=ngene;meta['mitochondrial_fraction']=mt/np.maximum(total,1)
ctrl=meta.Treatment.eq('Ctrl');m=float(meta.loc[ctrl,'DDR34'].median());mad=float(np.median(np.abs(meta.loc[ctrl,'DDR34']-m)))
meta['DDR_z']=(meta.DDR34-m)/(1.4826*mad);meta['DDR_high']=meta.DDR_z>2
meta.reset_index(names='barcode').to_csv(ROOT/'scored_rna.csv.gz',index=False,float_format='%.17g')
facts={'n_cells':len(bar),'n_genes':len(symbols),'nnz':nnz,'n_ctrl':int(ctrl.sum()),'frozen_reference':{'median':m,'MAD':mad,'cutoff_z2':m+2*1.4826*mad},'coverage':{name:sum(lookup[g] is not None for g in members) for name,members in sets.items()},'full_GEO_features_vs_source_metadata_max_difference':float(np.max(np.abs(ngene-meta.nFeature_RNA))), 'full_GEO_count_vs_source_metadata_max_difference':float(np.max(np.abs(total-meta.nCount_RNA))), 'full_GEO_count_vs_source_metadata_median_ratio':float(np.median(total/meta.nCount_RNA)),'normalization':'log1p(10000*integer_count/full deposited RNA column sum)','identity_table_sha256':identity['source_table_sha256'],'original_matrix_shape':[nrow,ncol,nnz]}
(ROOT/'rna_score_definitions.json').write_text(json.dumps(facts,indent=2)+'\n')
print(json.dumps(facts,indent=2),flush=True)
