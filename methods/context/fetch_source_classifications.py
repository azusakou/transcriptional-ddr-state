#!/usr/bin/env python3
"""Retrieve only classification metadata from original public MIX-Seq ZIPs.

HTTP byte ranges avoid downloading expression matrices. The archive catalog is
the publisher-linked Figshare article 10298696. Abort if Range is not honored.
"""
from pathlib import Path
import io, json, struct, zipfile, zlib, hashlib
import requests

HERE = Path(__file__).resolve().parent


def get_range(url, start, end):
    response = requests.get(
        url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=40
    )
    if response.status_code != 206:
        response.close()
        raise RuntimeError(
            f"Source did not honor byte range: HTTP {response.status_code}"
        )
    data = response.content
    if len(data) != end - start + 1:
        raise ValueError("Unexpected byte-range length")
    return data


def extract(file):
    url = file["download_url"]
    size = file["size"]
    tail = get_range(url, max(0, size - 65557), size - 1)
    end = tail.rfind(b"PK\x05\x06")
    if end < 0:
        raise ValueError("No ZIP end record")
    fields = struct.unpack_from("<4s4H2LH", tail, end)
    cd_size, cd_offset = fields[5:7]
    central = get_range(url, cd_offset, cd_offset + cd_size - 1)
    infos = zipfile.ZipFile(io.BytesIO(central + tail[end:])).infolist()
    matches = [x for x in infos if x.filename.endswith("classifications.csv")]
    if len(matches) != 1:
        raise ValueError("Ambiguous classification member")
    info = matches[0]
    # ZipFile adjusts offsets for our synthetic central-directory-only file.
    actual_header_offset = info.header_offset + cd_offset
    local = get_range(url, actual_header_offset, actual_header_offset + 29)
    header = struct.unpack("<4s5H3L2H", local)
    offset = actual_header_offset + 30 + header[-2] + header[-1]
    packed = get_range(url, offset, offset + info.compress_size - 1)
    content = zlib.decompress(packed, -15) if info.compress_type == 8 else packed
    assert len(content) == info.file_size and zlib.crc32(content) == info.CRC
    name = file["name"].replace(".zip", "_classifications.csv")
    (HERE / name).write_bytes(content)
    return dict(
        file=name,
        archive=file["name"],
        archive_url=url,
        archive_bytes=size,
        member=info.filename,
        member_uncompressed_bytes=len(content),
        member_crc32=info.CRC,
        sha256=hashlib.sha256(content).hexdigest(),
        transfer="four HTTP byte ranges; expression matrix not downloaded",
    )


def main():
    record = json.loads((HERE / "figshare_record_10298696.json").read_text())
    wanted = [
        "DMSO_6hr_expt1.zip",
        "DMSO_24hr_expt1.zip",
        "Untreated_6hr_expt1.zip",
        "DMSO_6hr_expt3.zip",
        "DMSO_24hr_expt3.zip",
        "DMSO_expt10.zip",
        "Trametinib_24hr_expt1.zip",
        "Trametinib_24hr_expt3.zip",
    ]
    manifest = (
        json.loads((HERE / "source_classification_manifest.json").read_text())
        if (HERE / "source_classification_manifest.json").exists()
        else []
    )
    for name in wanted:
        prior = next((x for x in manifest if x["archive"] == name), None)
        if (
            prior
            and (HERE / prior["file"]).exists()
            and hashlib.sha256((HERE / prior["file"]).read_bytes()).hexdigest()
            == prior["sha256"]
        ):
            continue
        source = next(f for f in record["files"] if f["name"] == name)
        item = extract(source)
        manifest.append(item)
        print(name, item["member_uncompressed_bytes"], flush=True)
        (HERE / "source_classification_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )


if __name__ == "__main__":
    main()
