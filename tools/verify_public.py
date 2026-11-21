"""Verify the integrity of the published evidence snapshot."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "provenance/public_snapshot.json"
LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def staged_blobs(names):
    requests = ("".join(f":{name}\n" for name in names)).encode("utf8")
    result = subprocess.run(["git", "cat-file", "--batch"], cwd=ROOT,
                            input=requests, capture_output=True, check=True)
    with io.BytesIO(result.stdout) as stream:
        for name in names:
            header = stream.readline().strip().split()
            assert len(header) == 3 and header[1] == b"blob", name
            yield stream.read(int(header[2]))
            assert stream.read(1) == b"\n", name


def verify(staged=False):
    inventory = json.loads(INVENTORY.read_bytes())
    files = inventory["files"]
    assert len(files) == inventory["file_count"]
    receipts = sum(item["role"] == "receipt" for item in files.values())
    assert receipts == inventory["run_manifest_count"]
    raw_dir = ROOT / "data/raw"
    assert all(p.suffix == ".json" for p in raw_dir.rglob("*") if p.is_file())
    python_count = 0
    indexed = staged_blobs(files) if staged else None
    for name, item in files.items():
        path = ROOT / name
        assert path.resolve().is_relative_to(ROOT), name
        data = path.read_bytes()
        expected = item["sha256"]
        assert len(data) == item["bytes"], name
        assert hashlib.sha256(data).hexdigest() == expected, name
        if path.suffix == ".py":
            ast.parse(data, filename=name)
            python_count += 1
        if indexed is not None:
            blob = next(indexed)
            if blob.startswith(LFS_PREFIX):
                assert f"oid sha256:{expected}".encode() in blob.splitlines(), name
                assert f"size {len(data)}".encode() in blob.splitlines(), name
            else:
                assert hashlib.sha256(blob).hexdigest() == expected, name
    if indexed is not None:
        assert next(indexed, None) is None
    print(f"Verified {len(files)} files, {receipts} run manifests and {python_count} Python sources.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="Verify against the Git index too")
    verify(parser.parse_args().staged)

