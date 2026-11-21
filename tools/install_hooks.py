"""Enable repository-local release policy and Git LFS hooks after a clone."""
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def main():
    top=subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=ROOT,text=True).strip()
    if Path(top).resolve()!=ROOT:raise SystemExit('Hook installation requires this project as the Git root')
    subprocess.run(['git','lfs','version'],cwd=ROOT,check=True)
    subprocess.run(['git','config','--local','core.hooksPath','.githooks'],cwd=ROOT,check=True)
    print('Enabled .githooks: staged release checks, commit subject validation, and Git LFS integration.')

if __name__=='__main__':main()
