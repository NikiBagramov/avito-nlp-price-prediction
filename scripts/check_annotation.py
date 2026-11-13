import json
import re
import subprocess
from common import ROOT,data
source=(ROOT/'annotation/index.html').read_text(encoding='utf-8')
js=re.search(r'<script>(.*?)</script>',source,re.S)[1]
dest=ROOT/'tmp';dest.mkdir(exist_ok=True);path=dest/'annotation.js';path.write_text(js,encoding='utf-8')
subprocess.run(['node','--check',str(path)],check=True)
sample=json.loads((ROOT/'data/annotations/review_100.json').read_text(encoding='utf-8'))
assert len(sample)==100 and all(set(r)=={'listing_id','text'} for r in sample)
d=data();assert set(r['listing_id'] for r in sample)<=set(d.loc[d.llm_subset & d.split.eq('test'),'listing_id'])
assert 'fetch(' not in js and 'XMLHttpRequest' not in js
print('Annotation JavaScript syntax and blind sample schema checked. Browser visual verification was unavailable in the in-app browser (localhost timeout).')
