#!/usr/bin/env python3
"""Rebuild fixed TP53 calls from the compact, source-attributed variant extract."""
from pathlib import Path
import argparse,json
import pandas as pd
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--records',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
plan=json.loads((a.records/'CCLE2019_classification_plan.json').read_text());samples=pd.read_csv(a.records/'CCLE2019_profiled_samples.csv');v=pd.read_csv(a.records/'CCLE2019_TP53_variants.csv.gz')
assert len(samples)==84 and samples.cell_line.is_unique and samples.CCLE_ID.is_unique and samples.TP53_profiled.all()
assert samples.sample_list_id.eq('ccle_broad_2019_sequenced').all()
assert v.molecularProfileId.eq(plan['profile']).all() and v.entrezGeneId.eq(plan['entrez_gene']).all() and v.sampleId.isin(samples.CCLE_ID).all()
positive=set(v.loc[v.mutationType.isin(plan['positive_classes']),'sampleId']);samples['TP53_raw_qualifying_call']=samples.CCLE_ID.isin(positive).astype(int)
curation=pd.read_csv(a.records/'TP53_literature_splice_curation.csv');assert curation.cell_line.is_unique and set(curation.cell_line)=={'NCIH2347'}
samples=samples.merge(curation[['cell_line','literature_splice_curation']],on='cell_line',how='left',validate='one_to_one');samples['literature_splice_curation']=samples.literature_splice_curation.eq(True);samples['TP53_curated_qualifying']=(samples.TP53_raw_qualifying_call.eq(1)|samples.literature_splice_curation).astype(int)
frozen=pd.read_csv(a.records/'curated_TP53_annotations.csv').set_index('cell_line');new=samples.set_index('cell_line')
for column in ['TP53_raw_qualifying_call','TP53_curated_qualifying']:assert new[column].sort_index().equals(frozen[column].sort_index())
assert new.TP53_raw_qualifying_call.sum()==57 and new.TP53_curated_qualifying.sum()==58
samples.to_csv(a.output/'rebuilt_TP53_annotations.csv',index=False);print('Exact 84/84 sample joins; 57 raw calls; 58 curated calls; no unknown annotations.')
