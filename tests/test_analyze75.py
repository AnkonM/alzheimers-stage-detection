"""Analyze 7.5 reader on SYNTHETIC files only (no OASIS data): explicit orientation, marker voxel per side,
both endiannesses, wrong sizes, refusal of derivative folders, converted output kept outside git trees."""
import struct

import nibabel as nib
import numpy as np
import pytest

from src.assessment.readers import analyze75 as A

SHAPE = (256, 256, 128)
SMALL = (8, 10, 6)


def write_analyze(stem, data, *, endian="<", orient=2, pixdim=(1.0, 1.0, 1.25), datatype=4, bitpix=16,
                  dims=None, sizeof=348, truncate=0):
    dims = tuple(dims or data.shape)
    h = bytearray(348)
    struct.pack_into(endian + "i", h, 0, sizeof)
    struct.pack_into(endian + "8h", h, 40, 3, *dims, 1, 1, 1, 1)
    struct.pack_into(endian + "2h", h, 70, datatype, bitpix)
    struct.pack_into(endian + "8f", h, 76, 0.0, *pixdim, 0, 0, 0, 0)
    h[252] = orient
    stem.with_suffix(".hdr").write_bytes(bytes(h))
    img = data.astype(np.dtype(np.int16).newbyteorder(endian)).tobytes(order="F")
    stem.with_suffix(".img").write_bytes(img[:len(img) - truncate])
    return stem


def marker_volume(shape, voxel):
    v = np.zeros(shape, np.int16)
    v[voxel] = 1000
    return v


def world_of(img, voxel):
    return (img.affine @ np.array([*voxel, 1.0]))[:3]


def test_orientation_is_required_no_default(tmp_path):
    s = write_analyze(tmp_path / "a", marker_volume(SMALL, (1, 1, 1)))
    with pytest.raises(NotImplementedError):
        A.load_analyze75(s, expected_shape=None)
    with pytest.raises(NotImplementedError):
        A.load_analyze75(s, orientation=None, expected_shape=None)
    with pytest.raises(NotImplementedError):
        A.convert_to_nifti(s, tmp_path / "o.nii.gz", orientation=None, expected_shape=None)


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("conv,codes", [("radiological", ("A", "S", "L")), ("neurological", ("A", "S", "R"))])
def test_both_conventions_and_endiannesses(tmp_path, endian, conv, codes):
    vox = (2, 3, 1)                                   # a marker near the low end of the slice axis
    s = write_analyze(tmp_path / "a", marker_volume(SMALL, vox), endian=endian)
    img = A.load_analyze75(s, orientation=conv, expected_shape=None)
    assert nib.aff2axcodes(img.affine) == codes
    d = np.asarray(img.dataobj)
    assert d.shape == SMALL and d.dtype == np.int16 and int(d[vox]) == 1000 and int(d.sum()) == 1000
    np.testing.assert_allclose(img.header.get_zooms()[:3], (1.0, 1.0, 1.25))


def test_marker_ends_up_on_opposite_sides_under_the_two_conventions(tmp_path):
    low, high = (4, 4, 0), (4, 4, SMALL[2] - 1)
    for vox in (low, high):
        s = write_analyze(tmp_path / "m", marker_volume(SMALL, vox))
        r = world_of(A.load_analyze75(s, orientation="radiological", expected_shape=None), vox)
        n = world_of(A.load_analyze75(s, orientation="neurological", expected_shape=None), vox)
        assert r[0] == pytest.approx(-n[0]) and r[0] != 0          # mirror images about the midline
        assert r[1] == pytest.approx(n[1]) and r[2] == pytest.approx(n[2])   # A-P and S-I unchanged
    # low slice index: right of midline under 'radiological' (slice index increases towards patient left)
    s = write_analyze(tmp_path / "m", marker_volume(SMALL, low))
    assert world_of(A.load_analyze75(s, orientation="radiological", expected_shape=None), low)[0] > 0
    assert world_of(A.load_analyze75(s, orientation="neurological", expected_shape=None), low)[0] < 0


def test_explicit_axis_codes_override_and_validation(tmp_path):
    s = write_analyze(tmp_path / "a", marker_volume(SMALL, (1, 1, 1)), orient=0)
    img = A.load_analyze75(s, orientation="PIL", expected_shape=None)     # explicit codes work for any orient
    assert nib.aff2axcodes(img.affine) == ("P", "I", "L")
    with pytest.raises(A.RefusedInput):                                    # named convention needs orient 2
        A.load_analyze75(s, orientation="radiological", expected_shape=None)
    for bad in ("RAS?", "RLA", "AAS", "xyz", "", 5):
        with pytest.raises(ValueError):
            A.load_analyze75(write_analyze(tmp_path / "b", marker_volume(SMALL, (1, 1, 1))), orientation=bad,
                             expected_shape=None)


def test_wrong_sizes_refused(tmp_path):
    ok = marker_volume(SMALL, (1, 1, 1))
    with pytest.raises(A.RefusedInput):          # dims differ from the OASIS raw shape (default expected_shape)
        A.load_analyze75(write_analyze(tmp_path / "a", ok), orientation="radiological")
    with pytest.raises(A.RefusedInput):          # truncated .img
        A.load_analyze75(write_analyze(tmp_path / "b", ok, truncate=2), orientation="radiological", expected_shape=None)
    with pytest.raises(A.RefusedInput):          # header claims bigger dims than the .img holds
        A.load_analyze75(write_analyze(tmp_path / "c", ok, dims=(8, 10, 7)), orientation="radiological",
                         expected_shape=None)
    with pytest.raises(A.RefusedInput):          # not an Analyze header
        A.load_analyze75(write_analyze(tmp_path / "d", ok, sizeof=540), orientation="radiological", expected_shape=None)
    with pytest.raises(A.RefusedInput):          # bitpix mismatch
        A.load_analyze75(write_analyze(tmp_path / "e", ok, bitpix=32), orientation="radiological", expected_shape=None)
    with pytest.raises(A.RefusedInput):          # unsupported data type
        A.load_analyze75(write_analyze(tmp_path / "f", ok, datatype=128), orientation="radiological", expected_shape=None)
    with pytest.raises(A.RefusedInput):          # non-positive voxel size
        A.load_analyze75(write_analyze(tmp_path / "g", ok, pixdim=(1.0, 0.0, 1.25)), orientation="radiological",
                         expected_shape=None)


def test_exact_oasis_raw_shape_is_accepted(tmp_path):
    big = np.zeros(SHAPE, np.int16)
    big[5, 6, 7] = 321
    img = A.load_analyze75(write_analyze(tmp_path / "raw", big), orientation="neurological")
    assert img.shape == SHAPE and int(np.asarray(img.dataobj)[5, 6, 7]) == 321


@pytest.mark.parametrize("folder", ["PROCESSED", "processed", "FSL_SEG", "fsl_seg"])
def test_derivative_folders_refused(tmp_path, folder):
    d = tmp_path / "case" / folder / "MPRAGE" / "T88_111"
    d.mkdir(parents=True)
    s = write_analyze(d / "scan", marker_volume(SMALL, (1, 1, 1)))
    with pytest.raises(A.RefusedInput):
        A.load_analyze75(s, orientation="radiological", expected_shape=None)
    with pytest.raises(A.RefusedInput):
        A.load_analyze75(s.with_suffix(".img"), orientation="radiological", expected_shape=None)


def test_accepts_hdr_img_or_stem_paths(tmp_path):
    s = write_analyze(tmp_path / "a", marker_volume(SMALL, (1, 2, 3)))
    for p in (s, s.with_suffix(".hdr"), s.with_suffix(".img")):
        assert A.load_analyze75(p, orientation="radiological", expected_shape=None).shape == SMALL


def test_convert_writes_nifti_roundtrip_outside_git(tmp_path):
    s = write_analyze(tmp_path / "a", marker_volume(SMALL, (1, 2, 3)))
    out = A.convert_to_nifti(s, tmp_path / "x" / "case.nii.gz", orientation="neurological", expected_shape=None)
    back = nib.load(str(out))
    assert nib.aff2axcodes(back.affine) == ("A", "S", "R") and back.shape == SMALL
    assert int(np.asarray(back.dataobj)[1, 2, 3]) == 1000


def test_convert_refuses_outputs_inside_the_repo(tmp_path):
    from src.assessment.paths import REPO
    s = write_analyze(tmp_path / "a", marker_volume(SMALL, (1, 2, 3)))
    from src.assessment.pathguard import GuardError
    with pytest.raises(GuardError):
        A.convert_to_nifti(s, REPO / "var" / "should_not_exist.nii.gz", orientation="neurological", expected_shape=None)
    assert not (REPO / "var" / "should_not_exist.nii.gz").exists()


# ---------------------------------------------------------------- oasis1_primary / oasis1_mirrored
PRIMARY = np.array([[0, 0, -1.25, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
MIRRORED = np.array([[0, 0, 1.25, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)


@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("name,sform,codes", [("oasis1_primary", PRIMARY, ("A", "S", "L")),
                                              ("oasis1_mirrored", MIRRORED, ("A", "S", "R"))])
def test_oasis1_conventions_are_exactly_the_documented_sform(tmp_path, endian, name, sform, codes):
    s = write_analyze(tmp_path / "a", marker_volume(SMALL, (2, 3, 1)), endian=endian)
    img = A.load_analyze75(s, orientation=name, expected_shape=None)
    np.testing.assert_array_equal(img.affine, sform)                 # exact rows, zero translation
    assert nib.aff2axcodes(img.affine) == codes
    d = np.asarray(img.dataobj)
    assert int(d[2, 3, 1]) == 1000 and int(d.sum()) == 1000 and d.dtype == np.int16
    np.testing.assert_allclose(img.get_qform(), sform, atol=1e-6)    # qform and sform agree with the affine
    np.testing.assert_allclose(img.get_sform(), sform, atol=1e-6)


def test_oasis1_marker_voxel_lands_on_each_side(tmp_path):
    """world(i, j, k) = (-1.25 k, i, j) for the primary; the mirrored convention flips only the left-right sign."""
    for vox in ((3, 4, 0), (3, 4, 1), (3, 4, SMALL[2] - 1)):
        s = write_analyze(tmp_path / "m", marker_volume(SMALL, vox))
        p = world_of(A.load_analyze75(s, orientation="oasis1_primary", expected_shape=None), vox)
        m = world_of(A.load_analyze75(s, orientation="oasis1_mirrored", expected_shape=None), vox)
        np.testing.assert_allclose(p, (-1.25 * vox[2], vox[0], vox[1]))
        np.testing.assert_allclose(m, (1.25 * vox[2], vox[0], vox[1]))
    lo, hi = (3, 4, 0), (3, 4, SMALL[2] - 1)                         # increasing k moves towards patient LEFT (x falls)
    s = write_analyze(tmp_path / "m", marker_volume(SMALL, hi))
    assert world_of(A.load_analyze75(s, orientation="oasis1_primary", expected_shape=None), hi)[0] < \
        world_of(A.load_analyze75(s, orientation="oasis1_primary", expected_shape=None), lo)[0]


def test_primary_has_the_rotation_and_scale_of_the_legacy_radiological_convention(tmp_path):
    s = write_analyze(tmp_path / "a", marker_volume(SMALL, (1, 1, 1)))
    p = A.load_analyze75(s, orientation="oasis1_primary", expected_shape=None).affine
    r = A.load_analyze75(s, orientation="radiological", expected_shape=None).affine
    np.testing.assert_array_equal(p[:3, :3], r[:3, :3])
    assert not np.allclose(p[:3, 3], r[:3, 3])                       # only the translation differs
    n = A.load_analyze75(s, orientation="neurological", expected_shape=None).affine
    np.testing.assert_array_equal(A.load_analyze75(s, orientation="oasis1_mirrored", expected_shape=None).affine[:3, :3], n[:3, :3])


def test_oasis1_refusals(tmp_path):
    ok = marker_volume(SMALL, (1, 1, 1))
    for name in ("oasis1_primary", "oasis1_mirrored"):
        with pytest.raises(A.RefusedInput):                          # defined for orient code 2 only
            A.load_analyze75(write_analyze(tmp_path / "a", ok, orient=0), orientation=name, expected_shape=None)
        with pytest.raises(A.RefusedInput):                          # defined for 1 x 1 x 1.25 mm only
            A.load_analyze75(write_analyze(tmp_path / "b", ok, pixdim=(1.0, 1.0, 1.0)), orientation=name, expected_shape=None)
        with pytest.raises(A.RefusedInput):
            A.load_analyze75(write_analyze(tmp_path / "c", ok, pixdim=(1.1, 1.0, 1.25)), orientation=name, expected_shape=None)
        with pytest.raises(A.RefusedInput):                          # derivative folders stay refused
            d = tmp_path / "PROCESSED"
            d.mkdir(exist_ok=True)
            A.load_analyze75(write_analyze(d / "x", ok), orientation=name, expected_shape=None)
        with pytest.raises(NotImplementedError):                     # and there is still no default
            A.load_analyze75(write_analyze(tmp_path / "e", ok), orientation=None, expected_shape=None)
    assert set(A.NAMED) == {"oasis1_primary", "oasis1_mirrored", "radiological", "neurological"}


@pytest.mark.parametrize("name", ["oasis1_primary", "oasis1_mirrored"])
def test_nibabel_and_itk_read_the_written_nifti_identically(tmp_path, name):
    ants = pytest.importorskip("ants")
    s = write_analyze(tmp_path / "a", marker_volume(SMALL, (2, 3, 1)))
    out = A.convert_to_nifti(s, tmp_path / "o" / "case.nii.gz", orientation=name, expected_shape=None)
    nb = nib.load(str(out))
    a = ants.image_read(str(out))
    assert a.shape == nb.shape
    direction, spacing, origin = np.asarray(a.direction).reshape(3, 3), np.asarray(a.spacing), np.asarray(a.origin)
    for vox in ((0, 0, 0), (2, 3, 1), (7, 9, 5), (4, 0, 5)):
        ras = (nb.affine @ np.array([*vox, 1.0]))[:3]
        lps = origin + direction @ (spacing * np.array(vox))
        np.testing.assert_allclose(lps, (-ras[0], -ras[1], ras[2]), atol=1e-4)    # RAS -> LPS
    assert np.asarray(nb.dataobj)[2, 3, 1] == 1000 and a.numpy()[2, 3, 1] == 1000
