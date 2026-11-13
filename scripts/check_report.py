import json
from pathlib import Path
from pypdf import PdfReader
root=Path(__file__).resolve().parents[1]
r=PdfReader(root/'output/pdf/avito_nlp_report.pdf')
text='\n'.join(p.extract_text() for p in r.pages)
for heading in ['Abstract','1. Introduction','Team members and responsibilities','2. Related Work','3. Model Description','4. Dataset','5. Experiments','6. Results','7. Conclusion','References']:
    assert heading in text,heading
assert len(r.pages)==10
assert all(p.extract_text().strip() for p in r.pages)
config=json.loads((root/'project.json').read_text())
for author in config['authors']:assert author in text,author
if config['repository_url']:assert config['repository_url'] in text
else:assert 'repository URL has not yet been configured' in text
print('PDF text, 10 nonempty pages, template sections, authors and repository status checked. Visual review is recorded separately.')
