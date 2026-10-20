"""Verify the published file snapshot using standard-library SHA256."""
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parent
items=json.loads((root/'PACKAGE_SHA256.json').read_text(encoding='utf-8'))
failed=[]
for name,expected in items.items():
    p=root/name
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
        failed.append(name)
if failed:
    print('Missing or changed files:',*failed,sep='\n')
    sys.exit(1)
print(f'PASS: {len(items)} packaged files verified.')
