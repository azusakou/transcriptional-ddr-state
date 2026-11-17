from pathlib import Path
import os
import csv,hashlib,json,openpyxl
ROOT=Path(os.environ.get('DDR_LESION_ROOT',Path(__file__).parent))

def merge(v):
    out=[]
    for chrom,start,end in sorted(v):
        if out and out[-1][0]==chrom and start<=out[-1][2]:out[-1][2]=max(end,out[-1][2])
        else:out.append([chrom,start,end])
    return out

w=openpyxl.load_workbook(ROOT/'source_supp3.xlsx',read_only=True,data_only=True)
sheet=w['Supplementary_Table2'];it=sheet.iter_rows(values_only=True);next(it);heads=next(it)
rows=[dict(zip(heads,row)) for row in it if row[0]]
assert len(rows)==64794 and len({x['Barcodes'] for x in rows})==64794
for row in rows:
    row['capture_library']=row['Barcodes'].split('_')[0]
    row['barcode_tube']=row['Barcodes'].split(':')[-1]
    expected=['Ctrl','0hr','2hr','6hr','24hr','48hr'][(int(row['barcode_tube'])-1)//2]
    assert row['Treatment']==expected
with (ROOT/'source_cell_metadata.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
peaks={}
for sheet in w:
    if sheet.title.startswith('Supplementary_Table3_'):
        label=sheet.title.removeprefix('Supplementary_Table3_')
        v=[]
        for x in sheet.iter_rows(values_only=True):
            if isinstance(x[0],str) and x[0].startswith('chr') and isinstance(x[1],(int,float)):
                chrom=x[0]
                if chrom not in ['chr'+str(k) for k in range(1,23)]:continue
                v.append((chrom,int(x[1]),int(x[2])))
        peaks[label]=merge(v)
peaks['induced_union']=merge([v for k,vs in peaks.items() if k!='conserved' for v in vs])
with (ROOT/'fixed_damage_regions.bed').open('w') as f:
    f.write('chrom\tstart\tend\tset\n')
    for label,v in peaks.items():
        for ch,st,en in v:f.write(f'{ch}\t{st}\t{en}\t{label}\n')
facts={'paper_doi':'10.1038/s41592-025-02632-3','GEO':'GSE268554','source_workbook_url':'https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-025-02632-3/MediaObjects/41592_2025_2632_MOESM3_ESM.xlsx','source_workbook_sha256':hashlib.sha256((ROOT/'source_supp3.xlsx').read_bytes()).hexdigest(),'author_code_commit':'2efce21e45045810a7e4473d1094a51089785781','n_cells':len(rows),'capture_libraries':sorted({r['capture_library'] for r in rows}),'physical_regions':{k:{'merged_autosomal_regions':len(v),'basepairs':sum(en-st for _,st,en in v)} for k,v in peaks.items()},'primary_endpoint':'Fraction of autosomal lesion-associated unique DNA fragments overlapping the frozen published union of induced regions','interpretation':'Capture-normalized relative localization of lesion-associated signal, not absolute lesions per cell or repair capacity','replication':'Two initial barcode tubes per condition and multiple capture libraries are technical sample partitions; independent culture replication not established.'}
facts.update({'barcode_collision_rate_source_estimate':0.114,'source_cell_line':'HeLa S3'})
existing=ROOT/'source_facts.json';recorded=json.loads(existing.read_text()) if existing.exists() else {};recorded.update(facts)
existing.write_text(json.dumps(recorded,indent=2)+'\n')
print(json.dumps(facts,indent=2))
