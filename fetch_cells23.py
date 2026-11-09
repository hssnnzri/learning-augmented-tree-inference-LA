"""Fetch the separately hosted 23-cell matrices without tracking them in this repository."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent


def fetch(source, destination):
    manifest = json.loads((HERE / "data/calibration_manifest.json").read_text())
    source = source or manifest["source_url"]
    destination.mkdir(parents=True, exist_ok=True)
    # Verify both inputs before replacing either destination file.
    contents = {}
    for name, expected in manifest["files"].items():
        if urlsplit(source).scheme in {"https", "http"}:
            with urlopen(source.rstrip("/") + "/" + name, timeout=60) as response:
                content = response.read()
        else:
            content = (Path(source).expanduser() / name).read_bytes()
        if len(content) != expected["bytes"] or hashlib.sha256(content).hexdigest() != expected["sha256"]:
            raise ValueError(f"Checksum or size mismatch for {name}; existing files were not changed")
        contents[name] = content
    for name, content in contents.items():
        with tempfile.NamedTemporaryFile(dir=destination, delete=False) as staged:
            staged.write(content)
            staged_path = Path(staged.name)
        try:
            os.replace(staged_path, destination / name)
        finally:
            staged_path.unlink(missing_ok=True)
        print(f"Verified and saved {destination / name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="Raw HTTP(S) file directory or local temporary data repository")
    parser.add_argument("--dest", type=Path, default=HERE / "data", help="Destination directory (default: data/)")
    args = parser.parse_args()
    try:
        fetch(args.source, args.dest)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Could not fetch the 23-cell data: {error}\n")


if __name__ == "__main__":
    main()
