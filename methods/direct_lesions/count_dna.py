#!/usr/bin/env python3
"""Count capture-normalized lesion-region overlaps, preserving paired cell IDs."""
from pathlib import Path
import os
import argparse,io,json,time,zlib
import numpy as np
import pandas as pd

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prefix',action='store_true');parser.add_argument('--follow',action='store_true');parser.add_argument('--output',type=Path);args=parser.parse_args()
    root=Path(os.environ.get('DDR_LESION_ROOT',Path(__file__).parent))
    out=args.output or root/'dna_counts.csv.gz'
    meta=pd.read_csv(root/'source_cell_metadata.csv');ids=meta.Barcodes.tolist();lookup={x:i for i,x in enumerate(ids)}
    regions=pd.read_csv(root/'fixed_damage_regions.bed',sep='\t');sets=list(regions['set'].unique())
    intervals={}
    for label in sets:
        for chrom,g in regions[regions['set'].eq(label)].groupby('chrom'):
            g=g.sort_values('start');intervals[label,chrom]=(g.start.to_numpy(),g.end.to_numpy())
    counts=np.zeros((len(ids),4+len(sets)),np.int64)
    names=['DNA_fragments_all','DNA_fragments_autosomal','DNA_fragments_mito','DNA_fragment_bases']+['DNA_'+x for x in sets]
    auto={'chr'+str(i) for i in range(1,23)}
    infile=root/'dna.gz';limit=infile.stat().st_size
    if args.prefix:
        state=json.loads((root/'dna_download_state.json').read_text());done=set(state['done']);limit=0
        chunk=4*1024**2
        while limit in done:limit+=chunk
        limit=min(limit,state['bytes'])
    def read_download_state():
        for retry in range(100):
            try:return json.loads((root/'dna_download_state.json').read_text())
            except (json.JSONDecodeError,OSError):time.sleep(.05)
        raise RuntimeError('Download state could not be read consistently')
    seen_libraries=set();nlines=0;nused=0;start=time.time();compressed=0;decoder=zlib.decompressobj(31);remain=b''
    def consume(payload):
        nonlocal nlines,nused
        if not payload:return
        a=pd.read_csv(io.BytesIO(payload),sep='\t',header=None,names=['chrom','start','end','barcode','multiplicity','strand'],dtype={'chrom':str,'start':np.int64,'end':np.int64,'barcode':str,'multiplicity':np.int32,'strand':str})
        nlines+=len(a)
        assert a.multiplicity.eq(1).all(),'Source multiplicity differs from unique-fragment convention'
        unique,inv=np.unique(a.barcode.to_numpy(),return_inverse=True)
        index=np.array([lookup.get(x,-1) for x in unique],np.int32)[inv];keep=index>=0
        a=a.loc[keep];index=index[keep];nused+=len(a)
        if not len(a):return
        seen_libraries.update(a.barcode.str.split('_').str[0].unique())
        counts[:,0]+=np.bincount(index,minlength=len(ids))
        counts[:,3]+=np.bincount(index,weights=(a.end-a.start).to_numpy(),minlength=len(ids)).astype(np.int64)
        for chrom,g in a.groupby('chrom',sort=False):
            # The original DataFrame row index tracks the selected-cell positions.
            local=np.flatnonzero(a.chrom.to_numpy()==chrom);ci=index[local]
            if chrom=='chrM':counts[:,2]+=np.bincount(ci,minlength=len(ids))
            if chrom not in auto:continue
            counts[:,1]+=np.bincount(ci,minlength=len(ids))
            starts=g.start.to_numpy();ends=g.end.to_numpy()
            for j,label in enumerate(sets):
                if (label,chrom) not in intervals:continue
                rs,re=intervals[label,chrom]
                ix=np.searchsorted(rs,ends,side='left')-1;valid=ix>=0
                ix_safe=np.maximum(ix,0);overlap=valid&(re[ix_safe]>starts)
                counts[:,4+j]+=np.bincount(ci[overlap],minlength=len(ids))
    with infile.open('rb') as f:
        while compressed<limit:
            if args.follow:
                done=set(read_download_state()['done']);available=0
                while available in done:available+=4*1024**2
                available=min(available,limit)
                if available<=compressed:
                    time.sleep(3);continue
                max_read=min(2**20,available-compressed)
            else:max_read=2**20
            raw=f.read(min(max_read,limit-compressed));compressed+=len(raw)
            decoded=decoder.decompress(raw)
            while decoder.unused_data:
                unused=decoder.unused_data;assert decoder.eof;decoder=zlib.decompressobj(31);decoded+=decoder.decompress(unused)
            payload=remain+decoded
            cut=payload.rfind(b'\n')
            if cut<0:remain=payload;continue
            consume(payload[:cut+1]);remain=payload[cut+1:]
            if compressed%(64*2**20)==0:print('Progress',round(compressed/limit*100,1),'%',nlines,'fragments',round(time.time()-start,1),'s',sorted(seen_libraries),flush=True)
        if not args.prefix:
            assert decoder.eof,'Full source gzip did not finish cleanly'
            if remain:consume(remain+b'\n')
    result=pd.DataFrame(counts,columns=names);result.insert(0,'barcode',ids)
    result.to_csv(out,index=False,compression='gzip')
    fact={'prefix_only':args.prefix,'compressed_bytes_processed':compressed,'source_complete_gzip':decoder.eof,'total_fragment_rows':nlines,'paired_fragment_rows':nused,'seen_capture_libraries':sorted(seen_libraries),'n_paired_cells_detected':int((counts[:,0]>0).sum()),'elapsed_seconds':time.time()-start}
    (out.parent/(out.name+'.json')).write_text(json.dumps(fact,indent=2)+'\n')
    print(json.dumps(fact,indent=2),flush=True)

if __name__=='__main__':main()
