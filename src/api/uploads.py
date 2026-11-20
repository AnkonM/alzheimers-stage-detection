"""Upload checks: NIfTI only, size-capped while streaming, content sniffed (not just the file name)."""
import gzip
import pathlib

from fastapi import HTTPException, UploadFile

from . import config

NIFTI_SUFFIXES = (".nii", ".nii.gz")


def check_name(name: str | None) -> str:
    n = (name or "").lower()
    if not n.endswith(NIFTI_SUFFIXES):
        raise HTTPException(415, "Only NIfTI files (.nii or .nii.gz) are accepted")
    return ".nii.gz" if n.endswith(".nii.gz") else ".nii"


def sniff(path: pathlib.Path):
    """A NIfTI-1/2 header has sizeof_hdr 348 (or 540) in its first four bytes, gzip-wrapped or not."""
    with open(path, "rb") as f:
        head = f.read(2)
        f.seek(0)
        raw = gzip.open(path).read(4) if head == b"\x1f\x8b" else f.read(4)
    if int.from_bytes(raw, "little") not in (348, 540) and int.from_bytes(raw, "big") not in (348, 540):
        raise HTTPException(415, "The file is not a NIfTI volume")


async def save(upload: UploadFile, dest_dir: pathlib.Path) -> pathlib.Path:
    suffix = check_name(upload.filename)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"scan{suffix}"          # the client's file name is never used on disk
    size = 0
    try:
        with open(dest, "wb") as f:
            while chunk := await upload.read(1 << 20):
                size += len(chunk)
                if size > config.MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"Upload larger than {config.MAX_UPLOAD_BYTES // (1 << 20)} MB")
                f.write(chunk)
        sniff(dest)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return dest
