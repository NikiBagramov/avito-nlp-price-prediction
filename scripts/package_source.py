"""Package the releasable source only; gitignore defines private/local exclusions."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode('utf-8').split('\0')
paths=sorted(set(p for p in paths if p))
forbidden=('data/raw/','data/processed/','data/annotations/','cache/','.deps/','.git/','tmp/')
assert not any(p.startswith(forbidden) or '/models/' in p or p.endswith('.jsonl') or p=='COURSE_SUBMISSION.md' for p in paths)
dest=ROOT.parent/'avito_nlp_project_source.zip'
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in paths:
        if (ROOT/p).is_file():z.write(ROOT/p,'avito_nlp_project/'+p)
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None
    assert 'avito_nlp_project/output/pdf/avito_nlp_report.pdf' in z.namelist()
print(json.dumps({'path':str(dest),'files':len(paths),'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()},indent=2))
