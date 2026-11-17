#!/usr/bin/env python3
from pathlib import Path
import os
import sys,json,hashlib,time
import h5py,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate_extensions as ev
R=Path(os.environ['DDR_EXTENSION_OUTPUT']).resolve()
started=time.time()
meta,names,modules,scores,pairs,controls,treated,folds,regimes,D,G=ev.load()
groups=[];lookup={};pair_groups=[]
for c,d in zip(pairs.cell_line,pairs.drug):
 ci=controls[(c,d)];key=tuple(ci.tolist())
 if key not in lookup:lookup[key]=len(groups);groups.append(ci)
 pair_groups.append(lookup[key])
raw=Path(os.environ['DDR_EXTENSION_RAW'])
with h5py.File(raw) as f:
 genes=ev.core.read_col(f['var'],'_index').astype(str);x=f['X'];ptr=x['indptr'][:]
 nsource=int(x.attrs['shape'][0]);rowgroup=np.full(nsource,-1,np.int16)
 counts=np.zeros(len(groups),int)
 for j,ci in enumerate(groups):
  orig=meta.iloc[ci].original_source_row.to_numpy(int);assert (rowgroup[orig]<0).all();rowgroup[orig]=j;counts[j]=len(orig)
 fullcounts=np.zeros(nsource,float);fullcounts[meta.original_source_row.to_numpy(int)]=meta.ncounts.to_numpy(float)
 out=np.zeros((len(groups),len(genes)),float)
 for a in range(0,len(genes),256):
  b=min(a+256,len(genes));lo=int(ptr[a]);hi=int(ptr[b]);rows=x['indices'][lo:hi];data=x['data'][lo:hi]
  gid=np.repeat(np.arange(b-a,dtype=np.int16),np.diff(ptr[a:b+1]));gr=rowgroup[rows];keep=gr>=0
  v=np.log1p(data[keep].astype(float)/fullcounts[rows[keep]]*10000)
  key=gr[keep].astype(np.int64)*(b-a)+gid[keep]
  out[:,a:b]=np.bincount(key,weights=v,minlength=len(groups)*(b-a)).reshape(len(groups),b-a)/counts[:,None]
  if a%2048==0:print('GENES',a,'of',len(genes),'elapsed',time.time()-started,flush=True)
np.savez_compressed(R/'control_transcriptome.npz',means=out,genes=genes,pair_groups=np.array(pair_groups),group_cell_counts=counts)
# Verify identity of means for every frozen marker module against existing scores.
gm=json.loads((ev.INPUT/'gene_identity_map.json').read_text())['source_symbol_map'];definitions=json.loads((ev.INPUT/'gene_sets.json').read_text());gene_lookup={g:i for i,g in enumerate(genes)}
checks=[]
for name in names:
 if name not in definitions:continue
 gl=definitions[name]['defined'];idx=[gene_lookup[g if g in gene_lookup else gm.get(g,g)] for g in gl if (g if g in gene_lookup else gm.get(g,g)) in gene_lookup]
 expected=np.array([scores[ci,names.index(name)].mean() for ci in groups]);recalc=out[:,idx].mean(1)
 error=float(np.max(abs(expected-recalc)));checks.append(dict(module=name,n_genes=len(idx),max_error=error));assert error<1e-8,(name,error)
(R/'transcriptome_extraction_checks.json').write_text(json.dumps(dict(n_unique_baseline_populations=len(groups),n_baseline_cells=int(counts.sum()),n_source_genes=len(genes),pair_count=len(pair_groups),checks=checks,elapsed_seconds=time.time()-started),indent=2)+'\n')
print('DONE',out.shape,'sec',time.time()-started,flush=True)
