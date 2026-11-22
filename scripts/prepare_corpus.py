"""Fetch and verify the public-domain corpus used by the full-conversion pilot."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest_path = Path(__file__).resolve().parents[1] / "results/full-conversion/corpus.json"
    manifest = json.loads(manifest_path.read_text())
    with urlopen(manifest["source"], timeout=30) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("Corpus archive checksum mismatch")
    with ZipFile(io.BytesIO(archive)) as corpus:
        for name, checksum in manifest["sha256"].items():
            archive_name = manifest["archive_names"].get(name, Path(name).name)
            payload = corpus.read(f"gutenberg/{archive_name}")
            if hashlib.sha256(payload).hexdigest() != checksum:
                raise ValueError(f"Corpus checksum mismatch: {name}")
            destination = args.output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)


if __name__ == "__main__":
    main()
