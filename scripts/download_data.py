"""Download the public, user-selected Kaggle release; never execute included scrapers."""
import hashlib
import json
from pathlib import Path
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://www.kaggle.com/api/v1/datasets/download/veronikavyalkova/saint-petersburg-avito-listings'
def main():
    dest = ROOT / 'data/raw'
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / 'kaggle_release.zip'
    if not archive.exists():
        part = archive.with_suffix('.part')
        with requests.get(URL, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with part.open('wb') as stream:
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
        if not zipfile.is_zipfile(part):
            raise ValueError('The server did not return a ZIP. Use the normal Kaggle download interface.')
        part.replace(archive)
    expected='987359dd03e80fc524ee8c564de839ed914dd379b586e3ec63c97f96c86d6ce0'
    actual=hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError('Kaggle release checksum changed. Obtain the documented original version; do not silently reuse old splits/models with new source data.')
    files = []
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise ValueError('Unsafe archive member')
            if not member.is_dir():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(member))
                files.append({'name': member.filename, 'bytes': target.stat().st_size,
                              'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    provenance = {'source': URL, 'dataset_page': 'https://www.kaggle.com/datasets/veronikavyalkova/saint-petersburg-avito-listings',
                  'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'files': files}
    (ROOT / 'artifacts').mkdir(exist_ok=True)
    (ROOT / 'artifacts/data_provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(provenance, indent=2, ensure_ascii=False))
if __name__ == '__main__':
    main()
