"""Enforce one patch increment and substantive scientific records from Git's index."""
import argparse
from datetime import date
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SECTIONS=['Purpose','Changes','Scientific record','Validation','Limitations']
VERSION_RE=re.compile(r'0\.0\.(0|[1-9][0-9]*)\Z')
ENTRY_RE=re.compile(r'^## \[([^\]]+)\] - (\d{4}-\d{2}-\d{2})\s*$',re.MULTILINE)

def patch_number(version):
    match=VERSION_RE.fullmatch(version)
    if not match:raise ValueError('VERSION must be 0.0.N with no leading zeroes')
    return int(match[1])

def next_version(previous):return f'0.0.{patch_number(previous)+1}'

def validate(previous,current,changelog,subject=None,previous_changelog=None):
    expected=next_version(previous)
    patch_number(current)
    if current!=expected:raise ValueError(f'VERSION must advance exactly once: {previous} -> {expected}')
    entries=list(ENTRY_RE.finditer(changelog))
    if not entries or entries[0][1]!=current:
        raise ValueError('CHANGELOG latest entry must match staged VERSION')
    if sum(m[1]==current for m in entries)!=1:raise ValueError('CHANGELOG version must occur exactly once')
    date.fromisoformat(entries[0][2])
    section=changelog[entries[0].end():entries[1].start() if len(entries)>1 else len(changelog)]
    headings=list(re.finditer(r'^### (.+)\s*$',section,re.MULTILINE))
    if [m[1].strip() for m in headings]!=SECTIONS:
        raise ValueError('CHANGELOG requires exactly these ordered sections: '+', '.join(SECTIONS))
    for index,heading in enumerate(headings):
        body=section[heading.end():headings[index+1].start() if index+1<len(headings) else len(section)].strip()
        plain=re.sub(r'[\s*`#\-]','',body)
        if len(plain)<12 or re.search(r'(?im)^\s*[-*]?\s*(?:TODO|TBD|PENDING|N/A|待填|待补充)\s*[.!。]?\s*$',body):
            raise ValueError('CHANGELOG section needs substantive content: '+heading[1].strip())
    if previous_changelog is not None:
        old=list(ENTRY_RE.finditer(previous_changelog))
        if old:
            retained=changelog[entries[1].start():].strip() if len(entries)>1 else ''
            if retained!=previous_changelog[old[0].start():].strip():
                raise ValueError('Previously committed CHANGELOG entries must remain unchanged')
    if subject is not None:
        prefix=f'v{current}: '
        if not subject.startswith(prefix) or not subject[len(prefix):].strip():
            raise ValueError(f'Commit subject must begin {prefix!r} followed by a change description')
    return current

def git(*args,required=True):
    result=subprocess.run(['git',*args],cwd=ROOT,capture_output=True)
    if result.returncode:
        if required:raise ValueError('Git operation failed: '+' '.join(args))
        return None
    return result.stdout.decode('utf8')

def check_index(message_file=None):
    changed=set(git('diff','--cached','--name-only','--diff-filter=ACMRD').splitlines())
    missing={'VERSION','CHANGELOG.md'}-changed
    if missing:raise ValueError('Every commit must stage VERSION and CHANGELOG.md; missing: '+', '.join(sorted(missing)))
    previous=(git('show','HEAD:VERSION',required=False) or '0.0.0').strip()
    current=git('show',':VERSION').strip()
    changelog=git('show',':CHANGELOG.md')
    old=git('show','HEAD:CHANGELOG.md',required=False)
    subject=None
    if message_file:
        message=Path(message_file).read_text(encoding='utf8')
        subject=message.splitlines()[0] if message.splitlines() else ''
    return validate(previous,current,changelog,subject=subject,previous_changelog=old)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged',action='store_true',help='Validate staged release records (default)')
    parser.add_argument('--message-file')
    options=parser.parse_args()
    try:version=check_index(options.message_file)
    except (ValueError,OSError,UnicodeError) as error:
        print('Release policy rejected commit: '+str(error),file=sys.stderr);return 1
    print('Release policy passed: v'+version)
    return 0

if __name__=='__main__':raise SystemExit(main())
