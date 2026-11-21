"""Prepare the next patch/changelog together; does not stage or commit changes."""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess

from release_guard import next_version, validate, ENTRY_RE

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--title',required=True)
    parser.add_argument('--date',default=datetime.now(timezone(timedelta(hours=8))).date().isoformat())
    for key in ['purpose','changes','scientific-record','validation','limitations']:
        parser.add_argument('--'+key,action='append',required=True)
    args=parser.parse_args()
    if '\n' in args.title or not args.title.strip():parser.error('Title must be a nonempty single line')
    result=subprocess.run(['git','show','HEAD:VERSION'],cwd=ROOT,capture_output=True,text=True)
    if result.returncode:parser.error('Create the initial baseline before using this helper')
    previous=result.stdout.strip();current=(ROOT/'VERSION').read_text(encoding='utf8').strip()
    if current!=previous:parser.error('VERSION already differs from HEAD; do not prepare twice')
    version=next_version(previous)
    changelog=(ROOT/'CHANGELOG.md').read_text(encoding='utf8')
    parent=subprocess.run(['git','show','HEAD:CHANGELOG.md'],cwd=ROOT,capture_output=True,check=True).stdout.decode('utf8')
    sections=[('Purpose',args.purpose),('Changes',args.changes),('Scientific record',args.scientific_record),
              ('Validation',args.validation),('Limitations',args.limitations)]
    entry=f'## [{version}] - {args.date}\n\n'
    for name,records in sections:entry+='### '+name+'\n'+''.join('- '+value.strip()+'\n' for value in records)+'\n'
    first=ENTRY_RE.search(changelog)
    if not first:parser.error('Cannot locate previous changelog entry')
    updated=changelog[:first.start()]+entry+changelog[first.start():]
    validate(previous,version,updated,subject=f'v{version}: {args.title.strip()}',previous_changelog=parent)
    (ROOT/'VERSION').write_text(version+'\n',encoding='utf8')
    (ROOT/'CHANGELOG.md').write_text(updated,encoding='utf8')
    print(f'Prepared v{version}: {args.title.strip()}')
    print('Review changes, run recorded checks, then stage and commit with this exact version prefix.')

if __name__=='__main__':main()
