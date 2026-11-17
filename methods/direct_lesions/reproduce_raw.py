#!/usr/bin/env python3
"""Reconstruct fixed same-cell physical endpoints from verified public files."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--records',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--download',action='store_true');a=p.parse_args();a.data_root.mkdir(parents=True,exist_ok=True)
for f in a.records.iterdir():
 if f.suffix=='.json':shutil.copy2(f,a.data_root/f.name)
os.environ['DDR_LESION_ROOT']=str(a.data_root.resolve());code=Path(__file__).resolve().parent
if a.download:subprocess.run([sys.executable,str(code/'download_inputs.py')],check=True)
if a.output:
 for record in json.loads((a.records/'input_manifest.json').read_text()):
  source=a.data_root/record['filename']
  with source.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
  assert source.stat().st_size==record['bytes'] and sha==record['sha256'], 'Raw physical source checksum mismatch: '+source.name
 a.output.mkdir(parents=True,exist_ok=True)
 for name in ['prepare_sources.py','score_rna.py','count_dna.py']:subprocess.run([sys.executable,str(code/name)],check=True)
 subprocess.run([sys.executable,str(code/'analyse_pairs.py'),'--output',str(a.output)],check=True)
 subprocess.run([sys.executable,str(code/'sensitivity.py'),'--input',str(a.output/'paired_cells.csv.gz'),'--output',str(a.output)],check=True)
