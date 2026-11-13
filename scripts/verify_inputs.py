import hashlib
import json
from common import ROOT,data
def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()
p=json.loads((ROOT/'artifacts/data_provenance.json').read_text(encoding='utf-8'))
assert checksum(ROOT/'data/raw/kaggle_release.zip')==p['archive_sha256']
for row in p['files']:assert checksum(ROOT/'data/raw'/row['name'])==row['sha256'],row['name']
d=data()
assert len(d)==30775 and d.listing_id.nunique()==30775
assert d.split.value_counts().to_dict()=={'train':21718,'dev':4582,'test':4475}
print('Verified original archive, 3 extracted source checksums, distinct IDs and frozen split counts.')
