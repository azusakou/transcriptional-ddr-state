#!/usr/bin/env python3
"""Fetch immutable public inputs and verify the recorded byte counts and SHA-256."""
from pathlib import Path
import os
import hashlib,json,time,urllib.request
ROOT=Path(os.environ.get('DDR_LESION_ROOT',Path(__file__).parent))
for item in json.loads((ROOT/'input_manifest.json').read_text()):
    target=ROOT/item['filename'];part=target.with_suffix(target.suffix+'.part')
    if target.exists():
        with target.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        if target.stat().st_size==item['bytes'] and digest==item['sha256']:continue
        raise ValueError('Existing input fails recorded checksum: '+target.name)
    for attempt in range(10):
        offset=part.stat().st_size if part.exists() else 0
        try:
            request=urllib.request.Request(item['url'],headers={'Range':f'bytes={offset}-','User-Agent':'Paired-Damage-seq-reanalysis'})
            with urllib.request.urlopen(request,timeout=120) as response:
                if offset:
                    assert response.status==206 and response.headers['Content-Range'].startswith(f'bytes {offset}-')
                with part.open('ab' if offset else 'wb') as out:
                    for block in iter(lambda:response.read(2**20),b''):out.write(block)
            assert part.stat().st_size==item['bytes']
            with part.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
            assert digest==item['sha256'],'Input SHA-256 mismatch'
            part.replace(target);break
        except Exception:
            if attempt==9:raise
            time.sleep(min(60,5*(attempt+1)))
    print('Verified',target.name,flush=True)
