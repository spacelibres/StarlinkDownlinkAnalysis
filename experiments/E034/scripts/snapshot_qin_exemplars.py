"""Read a bounded selection from the author's range-addressable HDF5 archive."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import ssl
import sys
import urllib.request

import h5py
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context

URL = ("https://rnl-data.ae.utexas.edu/datastore/supplementaryMaterial/qin-starlink-pilots/"
       "exemplar-frames-data/decodedExemplarFrames_20251019T2214.mat")
ETAG = '"142cf697b-649e35686ef48"'
SIZE = 5415856507
PAGE = 4 * 1024 * 1024
GROUPS = (0, 130, 247, 260, 390, 520, 650, 780, 988)
SOFT_START = 988
SOFT_SLOTS = 48


class ArchiveReader(io.RawIOBase):
    def __init__(self, cache, use_unverified_tls):
        self.position = 0
        self.cache = cache
        self.pages = {}
        self.receipts = {}
        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT) if use_unverified_tls else ssl.create_default_context()
        if use_unverified_tls:
            self.context.check_hostname = False
            self.context.verify_mode = ssl.CERT_NONE

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        position = (0, self.position, SIZE)[whence] + offset
        if position < 0:
            raise ValueError("Negative archive seek")
        self.position = position
        return position

    def load_page(self, start):
        stop = min(start + PAGE, SIZE)
        path = self.cache / f"{start}-{stop-1}.bin"
        if path.exists():
            data = path.read_bytes()
            source = "local_range_cache"
        else:
            header = f"bytes {start}-{stop-1}/{SIZE}"
            request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{stop-1}", "If-Match": ETAG})
            with urllib.request.urlopen(request, context=self.context, timeout=35) as response:
                if response.status != 206 or response.headers.get("Content-Range") != header:
                    raise ValueError("Source did not return the requested interval")
                if response.headers.get("ETag") != ETAG:
                    raise ValueError("Source revision changed")
                data = response.read(PAGE + 1)
            if len(data) == stop - start:
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(data)
                temporary.replace(path)
            source = "http_206"
        if len(data) != stop - start:
            raise ValueError(f"Incorrect data length at archive offset {start}")
        self.receipts[start] = {"bytes": [start, stop], "sha256": hashlib.sha256(data).hexdigest(), "read_via": source}
        if len(self.pages) == 12:
            self.pages.pop(next(iter(self.pages)))
        self.pages[start] = data
        return data

    def readinto(self, buffer):
        total = min(len(buffer), max(0, SIZE - self.position))
        view = memoryview(buffer)
        copied = 0
        while copied < total:
            start = self.position // PAGE * PAGE
            page = self.pages.get(start)
            if page is None:
                page = self.load_page(start)
            offset = self.position - start
            count = min(total - copied, len(page) - offset)
            view[copied:copied+count] = page[offset:offset+count]
            copied += count
            self.position += count
        return copied


def as_complex(raw):
    result = np.empty(raw.shape, dtype=np.complex128)
    result.real = raw["real"]
    result.imag = raw["imag"]
    return result


def main():
    require_run_context()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-unverified-source", action="store_true",
                        help="Allow the pinned public university endpoint when local TLS validation fails")
    args = parser.parse_args()
    out = experiment_output("E034")
    out.mkdir(parents=True, exist_ok=True)
    cache = ROOT / ".cache" / "qin-mat-pages"
    cache.mkdir(parents=True, exist_ok=True)
    reader = ArchiveReader(cache, args.allow_unverified_source)
    with reader, h5py.File(reader, "r") as data:
        hard = data["yDataDec"]
        soft = data["yDataSoft"]
        if hard.shape != (1009, 301, 1024) or soft.shape != hard.shape:
            raise ValueError("The archived MATLAB axes have changed")
        frames = np.arange(hard.shape[0])
        filename = "".join(map(chr, data["filename"][...].ravel()))
        for start in GROUPS:
            selection = slice(start, start + 13)
            np.savez_compressed(out / f"qin_hard_{start}.npz", frame=frames[selection],
                                toa_s=data["TOAs"][0, selection], sinr_db=data["SINRs"][0, selection],
                                symbols=as_complex(hard[selection]))
            print("Archived hard window", start, flush=True)
        selection = slice(SOFT_START, SOFT_START + 13)
        np.savez_compressed(out / "qin_soft_988.npz", frame=frames[selection],
                            toa_s=data["TOAs"][0, selection], sinr_db=data["SINRs"][0, selection],
                            symbols=as_complex(soft[selection, :SOFT_SLOTS]))
    record = {"schema": "qin-exemplar-snapshot/v1", "url": URL, "etag": ETAG, "source_size": SIZE,
              "upstream_capture_name": filename, "hard_groups": list(GROUPS), "group_length": 13,
              "soft_first_frame": SOFT_START, "soft_first_slots": SOFT_SLOTS,
              "h5py_version": h5py.__version__,
              "network_tls_validated": not args.allow_unverified_source,
              "cache_transport_caveat": "Existing range cache was filled without TLS verification for this host.",
              "pages": [reader.receipts[p] for p in sorted(reader.receipts)],
              "native_dataset_order": ["exemplar_index", "ofdm_symbol_1_to_301", "fft_bin"],
              "credit": "Qin, Psiaki, Bowman and Humphreys. Data licensed CC BY 4.0 by authors."}
    (out / "qin_snapshot_source.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
