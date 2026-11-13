"""Single fresh-run entry point; GPU stages are deliberately sequential."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
def run(*args):
    print('RUN',*args,flush=True)
    subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--skip-download',action='store_true');ap.add_argument('--skip-generation-download',action='store_true');ap.add_argument('--no-report',action='store_true');a=ap.parse_args()
    if not a.skip_download:run('scripts/download_data.py')
    if not (ROOT/'data/raw/listings.csv').exists():run('scripts/read_workbook.py')
    run('scripts/prepare.py')
    if not (ROOT/'cache/embedding_model/model.safetensors').exists():run('scripts/download_embedding.py')
    if not a.skip_generation_download and not os.environ.get('GENERATION_MODEL_PATH') and not (ROOT/'cache/generation_model/config.json').exists():run('scripts/download_generation.py')
    run('scripts/llm.py','features')
    run('scripts/llm.py','features','--prompt-version','fewshot')
    run('scripts/llm.py','prices')
    run('scripts/embed.py')
    run('scripts/benchmark.py','--embeddings')
    run('scripts/benchmark.py','--subset','--embeddings','--llm')
    run('scripts/rich_oof.py')
    run('scripts/augment_rich.py')
    run('scripts/evaluate_llm_prices.py')
    run('scripts/refresh_test_metrics.py')
    run('scripts/analyze.py')
    run('scripts/verify_results.py')
    run('-m','unittest','discover','-s','tests','-v')
    if not a.no_report:run('scripts/build_report.py')
if __name__=='__main__':main()
