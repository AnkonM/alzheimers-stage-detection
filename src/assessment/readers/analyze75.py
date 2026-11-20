"""Analyze 7.5 (.hdr/.img) reader with an EXPLICIT orientation argument (never defaulted).

    load_analyze75(path, orientation="oasis1_primary" | "oasis1_mirrored" | "<3 axis codes>" | legacy name)

Named OASIS-1 conventions (header orient code 2 only, voxel size required to be exactly 1 x 1 x 1.25 mm):

    oasis1_primary  : voxel (i, j, k) -> RAS by the sform rows [0 0 -1.25 0], [1 0 0 0], [0 1 0 0], [0 0 0 1]:
                      i = posterior->anterior, j = inferior->superior, k = right->left (axis codes A, S, L), zero
                      translation. Source: the third-party recipe for converting raw OASIS-1 Analyze files to
                      NIfTI by A. M. Winkler (brainder.org, 2011-08-13). THIRD-PARTY DOCUMENTED, not stated by OASIS.
    oasis1_mirrored : the same with the sign of the k column reversed (k = left->right, axis codes A, S, R). A
                      pre-declared sensitivity (S6f), never the primary.

Legacy names used by the Stage B0 evidence tests, kept so that B0 stays reproducible: "radiological" (axis codes
ASL) and "neurological" (ASR), with the volume centred on the origin. "radiological" has the same rotation and
scale as oasis1_primary and differs only in translation.

The in-plane directions (A, S) and the left-right sign of the primary convention are as documented by the
third-party recipe; the Analyze 7.5 specification text read (the Mayo document) defines `orient` only as the slice
plane and a display flip and states no axis direction, so it neither confirms nor contradicts the recipe.

Refused: paths under PROCESSED/ or FSL_SEG/ (registered, masked or segmented derivatives are never model input),
wrong dimensions, a truncated or oversized .img, unsupported data types.
"""
import pathlib
import struct

import numpy as np

OASIS_RAW_SHAPE = (256, 256, 128)
CONVENTIONS = {"radiological": "ASL", "neurological": "ASR"}      # legacy (Stage B0), centred origin
OASIS1_VOXEL = (1.0, 1.0, 1.25)
OASIS1_SFORMS = {                                                    # exact sform rows, zero translation
    "oasis1_primary": np.array([[0, 0, -1.25, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float),
    "oasis1_mirrored": np.array([[0, 0, 1.25, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float),
}
NAMED = tuple(OASIS1_SFORMS) + tuple(CONVENTIONS)
REFUSED_DIRS = {"processed", "fsl_seg"}
DTYPES = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64}
SAGITTAL_UNFLIPPED = 2
_VEC = {"R": (1, 0, 0), "L": (-1, 0, 0), "A": (0, 1, 0), "P": (0, -1, 0), "S": (0, 0, 1), "I": (0, 0, -1)}


class RefusedInput(ValueError):
    """The file is not an acceptable raw Analyze scan."""


def _paths(path):
    p = pathlib.Path(path)
    if p.suffix.lower() in (".hdr", ".img"):
        stem = p.with_suffix("")
    else:
        stem = p
    hdr, img = stem.with_suffix(".hdr"), stem.with_suffix(".img")
    if any(part.lower() in REFUSED_DIRS for part in hdr.parts):
        raise RefusedInput("registered or segmented derivative folders (PROCESSED, FSL_SEG) are never model input")
    return hdr, img


def _named_affine(orientation, orient_code, pix):
    """The exact sform of an oasis1_* convention, or None when `orientation` is not one."""
    if not (isinstance(orientation, str) and orientation in OASIS1_SFORMS):
        return None
    if orient_code != SAGITTAL_UNFLIPPED:
        raise RefusedInput("named conventions are defined for header orient code 2 (sagittal) only; "
                           "pass explicit axis codes")
    if any(abs(a - b) > 1e-4 for a, b in zip(pix[:3], OASIS1_VOXEL)):
        raise RefusedInput("the oasis1 conventions are defined for 1 x 1 x 1.25 mm voxels only")
    return OASIS1_SFORMS[orientation].copy()


def _axis_codes(orientation, orient_code):
    if orientation is None:
        raise NotImplementedError("Analyze 7.5 has no reliable left-right sign: pass orientation=... explicitly "
                                  "(see the protocol, Stage B0)")
    if isinstance(orientation, str) and orientation in CONVENTIONS:
        if orient_code != SAGITTAL_UNFLIPPED:
            raise RefusedInput("named conventions are defined for header orient code 2 (sagittal) only; "
                               "pass explicit axis codes")
        return CONVENTIONS[orientation]
    if (isinstance(orientation, str) and len(orientation) == 3 and set(orientation) <= set(_VEC)
            and {c for c in orientation if c in "RL"} and {c for c in orientation if c in "AP"}
            and {c for c in orientation if c in "SI"}):
        return orientation
    raise ValueError("orientation must be 'oasis1_primary', 'oasis1_mirrored', a legacy name, or three axis codes "
                     "with one each of R/L, A/P and S/I")


def parse_header(raw: bytes):
    """348-byte Analyze header -> dict. Detects endianness from sizeof_hdr."""
    if len(raw) < 348:
        raise RefusedInput("header shorter than 348 bytes")
    for e in ("<", ">"):
        if struct.unpack(e + "i", raw[:4])[0] == 348:
            break
    else:
        raise RefusedInput("sizeof_hdr is not 348 (not an Analyze 7.5 header)")
    dim = struct.unpack(e + "8h", raw[40:56])
    datatype, bitpix = struct.unpack(e + "2h", raw[70:74])
    pixdim = struct.unpack(e + "8f", raw[76:108])
    vox_offset = struct.unpack(e + "f", raw[108:112])[0]
    return {"endian": e, "ndim": dim[0], "dim": tuple(dim[1:1 + max(dim[0], 0)]), "datatype": datatype,
            "bitpix": bitpix, "pixdim": tuple(pixdim[1:1 + max(dim[0], 0)]), "vox_offset": vox_offset,
            "orient": raw[252]}


def affine_from_codes(codes, voxel_sizes, shape):
    """RAS+ affine whose voxel axis i points towards codes[i], centred on the volume."""
    A = np.eye(4)
    for i, c in enumerate(codes):
        A[:3, i] = np.array(_VEC[c], float) * float(voxel_sizes[i])
    centre = (np.array(shape, float) - 1) / 2
    A[:3, 3] = -A[:3, :3] @ centre
    return A


def load_analyze75(path, *, orientation=None, expected_shape=OASIS_RAW_SHAPE):
    """Read a raw Analyze 7.5 volume into a nibabel Nifti1Image (in memory, int16 stays int16)."""
    import nibabel as nib
    if orientation is None:                      # checked first so a missing convention always raises
        raise NotImplementedError("Analyze 7.5 has no reliable left-right sign: pass orientation=... explicitly "
                                  "(see the protocol, Stage B0)")
    hdr_p, img_p = _paths(path)
    h = parse_header(hdr_p.read_bytes()[:348])
    dims = h["dim"]
    if h["ndim"] == 4 and dims[3] == 1:
        dims = dims[:3]
    if len(dims) != 3 or any(d <= 0 for d in dims):
        raise RefusedInput("not a 3D volume")
    if expected_shape is not None and tuple(dims) != tuple(expected_shape):
        raise RefusedInput("dimensions differ from the expected raw scan shape")
    if h["datatype"] not in DTYPES:
        raise RefusedInput("unsupported Analyze data type")
    dt = np.dtype(DTYPES[h["datatype"]]).newbyteorder(h["endian"])
    if h["bitpix"] != dt.itemsize * 8:
        raise RefusedInput("bitpix does not match the data type")
    n = int(np.prod(dims))
    if img_p.stat().st_size != n * dt.itemsize:
        raise RefusedInput(".img size does not match the header dimensions")
    pix = h["pixdim"][:3]
    if len(pix) != 3 or any(not np.isfinite(p) or p <= 0 for p in pix):
        raise RefusedInput("voxel sizes missing or not positive")
    aff = _named_affine(orientation, h["orient"], pix)
    if aff is None:
        aff = affine_from_codes(_axis_codes(orientation, h["orient"]), pix, dims)
    data = np.fromfile(img_p, dtype=dt, count=n).reshape(dims, order="F")
    data = np.ascontiguousarray(data.astype(dt.newbyteorder("="), copy=False))
    img = nib.Nifti1Image(data, aff)
    img.set_qform(aff, code=1)
    img.set_sform(aff, code=1)
    img.header.set_zooms(pix)
    return img


def convert_to_nifti(path, out_path, *, orientation=None, expected_shape=OASIS_RAW_SHAPE):
    """Convert one scan to a NIfTI file. The output path must lie outside every git work tree."""
    from ..pathguard import assert_outside_git
    if orientation is None:
        raise NotImplementedError("Analyze 7.5 has no reliable left-right sign: pass orientation=... explicitly")
    out = assert_outside_git(out_path, what="converted scan path")
    img = load_analyze75(path, orientation=orientation, expected_shape=expected_shape)
    out.parent.mkdir(parents=True, exist_ok=True)
    import nibabel as nib
    nib.save(img, str(out))
    return out
