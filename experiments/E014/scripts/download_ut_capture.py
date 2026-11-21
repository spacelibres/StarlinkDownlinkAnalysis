"""Download a hash-pinned published IQ asset without overwriting evidence."""
from pathlib import Path
from datetime import datetime,timezone
from urllib.request import Request,urlopen
from urllib.parse import urlparse
import hashlib
import json

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'project.json').is_file())
URL='https://github.com/misaka666/Starlink-Downlink-Analysis/releases/download/v2.0.0/Starlink_Downlink_IQ_250MSps.bin'
EXPECTED='c69e82022d45173cdcaf1a5fdc8a2b864e2d61e85ae819af54b16002044adae3'


def main():
    destination=ROOT/'data/raw/ut_250msps/Starlink_Downlink_IQ_250MSps.bin'
    manifest=ROOT/'provenance/ut_capture_import.json'
    if destination.exists() or manifest.exists():raise SystemExit('Refusing to overwrite imported evidence')
    destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_suffix('.download')
    if temporary.exists():raise SystemExit('Partial download exists; inspect before retry')
    digest=hashlib.sha256();size=0
    with urlopen(Request(URL,headers={'User-Agent':'StarlinkBeacon-research/0.0.11'}),timeout=60) as response,temporary.open('xb') as stream:
        host=urlparse(response.geturl()).hostname
        while block:=response.read(1048576):stream.write(block);digest.update(block);size+=len(block)
    actual=digest.hexdigest()
    if actual!=EXPECTED or size!=50000000:raise SystemExit('Hash/size mismatch; partial retained, not imported')
    temporary.rename(destination)
    record={'schema':'starlink-capture-import/v1','recorded_utc':datetime.now(timezone.utc).isoformat(),
            'source_url':URL,'redirected_host':host,'upstream_origin':'UT Austin Qin supplement exemplar250-257.zip, as attributed by release author',
            'metadata_url':'https://github.com/misaka666/Starlink-Downlink-Analysis/blob/main/data/metadata.json',
            'files':[{'snapshot':destination.relative_to(ROOT).as_posix(),'sha256':actual,'bytes':size}],
            'sample_rate_hz':250000000,'complex_samples':12500000,'dtype':'little-endian signed int16 interleaved I,Q',
            'scope':'Unmodified author release asset; independent RF from legacy24ms capture, not blind-to-literature validation. No new license asserted or remote publication.'}
    manifest.write_text(json.dumps(record,indent=2),encoding='utf8')
    print('Imported',size,'bytes SHA256',actual)


if __name__=='__main__':main()
