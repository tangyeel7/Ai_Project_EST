"""Fetch only DASIG's raw CSV files from the public Zenodo ZIP archive."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import struct
import time
import urllib.request
import zlib

URL = "https://zenodo.org/records/17660014/files/DASIG.zip?download=1"
TAIL_BYTES = 1_000_000


def get_range(start, end):
    last_error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=45) as response:
                content_range = response.headers.get("Content-Range", "")
                data = response.read()
                status = response.status
            if status != 206 or len(data) != end - start + 1:
                raise RuntimeError("Zenodo did not return the requested byte range.")
            return data, content_range
        except Exception as error:
            last_error = error
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f"Range {start}-{end} failed after retries: {last_error}")


def read_directory():
    request = urllib.request.Request(URL, headers={"Range": f"bytes=-{TAIL_BYTES}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        tail = response.read()
        content_range = response.headers.get("Content-Range", "")
        status = response.status
    if status != 206 or not content_range:
        raise RuntimeError("The Zenodo server must support byte-range requests for selective download.")
    tail_start = int(content_range.split()[1].split("-")[0])
    zip_size = int(content_range.rsplit("/", 1)[1])
    eocd_at = tail.rfind(b"PK\x05\x06")
    if eocd_at < 0:
        raise RuntimeError("Could not locate the ZIP end directory.")
    eocd = struct.unpack_from("<4s4H2LH", tail, eocd_at)
    count, directory_size, directory_offset = eocd[4], eocd[5], eocd[6]
    if directory_offset < tail_start:
        directory, _ = get_range(directory_offset, directory_offset + directory_size - 1)
    else:
        start = directory_offset - tail_start
        directory = tail[start:start + directory_size]

    entries = []
    offset = 0
    fmt = "<4s6H3L5H2L"
    while offset + 46 <= len(directory) and directory[offset:offset + 4] == b"PK\x01\x02":
        fields = struct.unpack_from(fmt, directory, offset)
        name_len, extra_len, comment_len = fields[10:13]
        name_bytes = directory[offset + 46:offset + 46 + name_len]
        encoding = "utf-8" if fields[3] & 0x800 else "cp437"
        name = name_bytes.decode(encoding)
        entries.append({"name": name, "method": fields[4], "crc": fields[7],
                        "compressed": fields[8], "size": fields[9], "local_offset": fields[16]})
        offset += 46 + name_len + extra_len + comment_len
    if len(entries) != count:
        raise RuntimeError(f"ZIP index is incomplete: found {len(entries)} of {count} entries.")
    return zip_size, entries


def selected(name):
    lower = name.lower()
    return (lower.endswith("_mimu.csv") or lower.endswith("_arduino.csv")
            or lower.endswith("_info.csv") or lower.endswith("/sub_info.csv")
            or lower.endswith("/readme.txt"))


def fetch_entry(entry, out_root):
    name = entry["name"]
    target = (out_root / Path(name)).resolve()
    if not target.is_relative_to(out_root.resolve()):
        raise RuntimeError(f"Unsafe archive path: {name}")
    if target.is_file() and target.stat().st_size == entry["size"]:
        crc = 0
        with target.open("rb") as saved:
            for block in iter(lambda: saved.read(1024 * 1024), b""):
                crc = zlib.crc32(block, crc)
        if crc & 0xffffffff == entry["crc"]:
            return name, -1
    local, _ = get_range(entry["local_offset"], entry["local_offset"] + 29)
    if local[:4] != b"PK\x03\x04":
        raise RuntimeError(f"Bad local ZIP header for {name}")
    fixed = struct.unpack_from("<4s5H3L2H", local)
    name_len, extra_len = fixed[9:11]
    start = entry["local_offset"] + 30 + name_len + extra_len
    compressed, _ = get_range(start, start + entry["compressed"] - 1)
    if entry["method"] == 8:
        content = zlib.decompress(compressed, -15)
    elif entry["method"] == 0:
        content = compressed
    else:
        raise RuntimeError(f"Unsupported ZIP compression method {entry['method']} for {name}")
    if len(content) != entry["size"] or zlib.crc32(content) & 0xffffffff != entry["crc"]:
        raise RuntimeError(f"Size or checksum mismatch for {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return name, len(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw/dasig")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    zip_size, all_entries = read_directory()
    entries = [item for item in all_entries if selected(item["name"])]
    total = sum(item["compressed"] for item in entries)
    print(f"DASIG archive: {zip_size / 1024**3:.2f} GiB; fetching {len(entries)} raw CSV/text files ({total / 1024**3:.2f} GiB compressed).")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(fetch_entry, entry, out_root) for entry in entries]
        for number, future in enumerate(as_completed(futures), 1):
            name, size = future.result()
            detail = "already verified" if size < 0 else f"{size:,} bytes"
            print(f"[{number}/{len(futures)}] {name} ({detail})", flush=True)
    print(f"DASIG raw CSV data ready in {out_root}")


if __name__ == "__main__":
    main()
