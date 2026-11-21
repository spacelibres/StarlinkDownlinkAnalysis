"""Copy fixed stored IQ ranges from the public StarLoc ZIP to an immutable snapshot."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import time
from urllib.request import Request, urlopen
from zipfile import ZIP_STORED, ZipFile


ROOT = Path(__file__).resolve().parents[3]
SOURCE = 'https://zenodo.org/records/19043854/files/starloc_dataset_obs_0.zip?download=1'
ZIP_BYTES = 42_362_931_803
TAIL_BYTES = 65_536
FS = 2_000_000
N = FS//10
STRIDE = 8
DIR = ROOT/'data/raw/starloc_obs0'
START_SECONDS = {
    'outside_author_windows': (8, 38, 98, 139, 245),
    'within_50_85': (55, 65, 75, 82),
    'within_160_195': (164, 172, 181, 190),
    'within_intersection_195_205': (200,),
    'within_205_230': (211, 219, 227),
}


def download(start, length):
    end = start+length-1
    if start < 0 or length <= 0 or end >= ZIP_BYTES:
        raise ValueError('Invalid source range')
    request = Request(SOURCE, headers={'Range': f'bytes={start}-{end}'})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=35) as reply:
                if reply.status != 206 or reply.headers.get('Content-Range') != f'bytes {start}-{end}/{ZIP_BYTES}':
                    raise ValueError('Server returned a different byte range')
                blob = reply.read(length+1)
                if len(blob) != length:
                    raise ValueError('Short or overlong HTTP range')
                return blob
        except OSError:
            if attempt == 3:
                raise
            time.sleep(1+attempt)


class ZipTail:
    """Supply the downloaded tail at its original coordinates to zipfile."""

    def __init__(self, tail):
        self.tail = tail
        self.position = 0

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        base = (0, self.position, ZIP_BYTES)[whence]
        self.position = base+offset
        return self.position

    def read(self, wanted=-1):
        start = self.position-(ZIP_BYTES-len(self.tail))
        if start < 0:
            raise OSError('Read outside the fetched ZIP64 directory')
        end = len(self.tail) if wanted < 0 else start+wanted
        out = self.tail[start:end]
        self.position += len(out)
        return out


def zip_members(tail):
    with ZipFile(ZipTail(tail)) as archive:
        members = [info for info in archive.infolist() if info.filename.endswith('.dat')]
        directory_offset = archive.start_dir
    members.sort(key=lambda entry: entry.filename)
    if [item.filename[-7:] for item in members] != ['rx0.dat', 'rx1.dat', 'rx2.dat']:
        raise ValueError('Unexpected IQ members')
    for rx, member in enumerate(members):
        if member.compress_type != ZIP_STORED or member.flag_bits & 1 or member.compress_size != member.file_size:
            raise ValueError('Direct addressing requires an unencrypted stored entry')
        header = download(member.header_offset, 30)
        signature, _, flags, method, _, _, _, _, _, namesize, extrasize = struct.unpack('<4s5H3I2H', header)
        if signature != b'PK\x03\x04' or flags != member.flag_bits or method != ZIP_STORED:
            raise ValueError('Inconsistent ZIP member local header')
        name = download(member.header_offset+30, namesize)
        if name.decode('ascii') != member.filename:
            raise ValueError('Inconsistent member path')
        start = member.header_offset+30+namesize+extrasize
        if start+member.file_size > directory_offset or member.file_size % STRIDE:
            raise ValueError('Member outside the addressable IQ region')
        yield {'receiver': rx, 'member': member.filename, 'header_offset': member.header_offset,
               'archive_data_offset': start, 'member_bytes': member.file_size, 'directory_crc32': f'{member.CRC:08x}'}


def window(task):
    lo = task['archive_offset']
    return download(lo, task['bytes'])


def main():
    if DIR.exists():
        raise SystemExit('The immutable StarLoc snapshot already exists')
    tail = download(ZIP_BYTES-TAIL_BYTES, TAIL_BYTES)
    members = list(zip_members(tail))
    requests = []
    for group, seconds in START_SECONDS.items():
        for start in seconds:
            for entry in members:
                begin = start*FS*STRIDE
                if begin+N*STRIDE > entry['member_bytes']:
                    raise ValueError('Sample range outside member')
                requests.append({'group': group, 'start_seconds': start, 'first_sample': start*FS,
                                 'sample_count': N, 'receiver': entry['receiver'], 'bytes': N*STRIDE,
                                 'archive_offset': entry['archive_data_offset']+begin})
    work = ROOT/'.cache'
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='starloc-', dir=work) as folder:
        snapshot = Path(folder)/'snapshot'
        snapshot.mkdir()
        digest = hashlib.sha256()
        with (snapshot/'selected_iq.bin').open('wb') as stream, ThreadPoolExecutor(max_workers=4) as pool:
            for task, data in zip(requests, pool.map(window, requests), strict=True):
                task['snapshot_offset'] = stream.tell()
                task['sha256'] = hashlib.sha256(data).hexdigest()
                stream.write(data)
                digest.update(data)
                print(task['start_seconds'], task['receiver'], task['sha256'], flush=True)
        metadata = {'schema': 'starlink-starloc-range-import/v1', 'source_url': SOURCE, 'source_zip_bytes': ZIP_BYTES,
                    'source_zip_tail_bytes': TAIL_BYTES, 'source_zip_tail_sha256': hashlib.sha256(tail).hexdigest(),
                    'sample_rate_hz': FS, 'tuned_center_hz': 11_325_000_000, 'iq_start_unix_utc': '1710285434.4520473',
                    'stored_format': 'interleaved little-endian float32 IQ as contiguous per-receiver windows',
                    'publisher_observation_windows_href': 'https://raw.githubusercontent.com/ConnectedSystemsLab/starloc_preproc/main/starlocProcessRawSignal/obs_periods.json',
                    'members': members, 'snapshot_sha256': digest.hexdigest(), 'windows': requests,
                    'scope': 'Partial remote byte ranges. Directory CRCs were copied, not verified against full members.'}
        (snapshot/'selection.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
        snapshot.rename(DIR)


if __name__ == '__main__':
    main()
