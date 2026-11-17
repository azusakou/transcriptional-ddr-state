#!/usr/bin/env python3
"""Retrieve the three pinned, small original source annotation files."""
from pathlib import Path
import hashlib, json, urllib.request

HERE = Path(__file__).resolve().parent


def main():
    records = json.loads((HERE / "downloaded_small_source_manifest.json").read_text())
    for item in records:
        target = HERE / item["name"]
        if (
            target.exists()
            and hashlib.sha256(target.read_bytes()).hexdigest() == item["sha256"]
        ):
            continue
        if item["bytes"] > 4_000_000:
            raise ValueError("Unexpected annotation size")
        with urllib.request.urlopen(item["source"], timeout=40) as response:
            content = response.read(item["bytes"] + 1)
        if (
            len(content) != item["bytes"]
            or hashlib.sha256(content).hexdigest() != item["sha256"]
        ):
            raise ValueError("Pinned source annotation size/checksum mismatch")
        target.write_bytes(content)
        print(item["name"], len(content))


if __name__ == "__main__":
    main()
