"""Spectral check for a regular lattice in an attribution map (run with any env that has numpy and nibabel).

    python tests/diagnostics/lattice_check.py MAP1.nii.gz MAP2.nii.gz ... [--scans SCAN1.nii.gz SCAN2.nii.gz ...]

MAPs are attribution volumes on the 96^3 grid (for example the overlay.nii.gz the app writes). For each map it prints, per axis,
the strongest spectral peaks of the mean-removed volume (k = cycles per 96 voxels) and their prominence, i.e. the power at k over
the median power of the neighbouring frequencies (k-5..k-2 and k+2..k+5). It then prints the correlation, inside the brain, of the
fine-scale component (period shorter than 12 voxels) between every pair of maps. A lattice that comes from the network's strides is
fixed in grid coordinates, so that correlation is high across different subjects; anatomy is not. --scans gives the same
fine-scale correlation for the subjects' own scans as the anatomical control.

Context (2026-11-04): the layer1 LayerCAM maps of the app's demo cases peak at k = 12 on all three axes (period 8 voxels, two layer1
cells). See analysis/explainability/lattice_note.md. Reads volumes only; no labels, no metrics.
"""
import argparse
import itertools

import nibabel as nib
import numpy as np


def axis_profiles(v, kmax=48):
    f = np.abs(np.fft.fftn(v - v.mean())) ** 2
    return [f.mean(axis=tuple(a for a in range(3) if a != ax))[: kmax + 1] for ax in range(3)]


def top_peaks(p, n=2):
    r = []
    for k in range(3, len(p) - 2):
        nb = np.r_[p[max(1, k - 5):k - 1], p[k + 2:k + 6]]
        r.append((k, float(p[k] / np.median(nb))))
    return sorted(r, key=lambda x: -x[1])[:n]


def fine_scale(v, min_period=12):
    f = np.fft.fftn(v)
    k2 = sum(np.fft.fftfreq(s)[tuple(slice(None) if a == i else None for a in range(3))] ** 2 for i, s in enumerate(v.shape))
    f[k2 < (1 / min_period) ** 2] = 0                         # drop everything coarser than min_period voxels
    return np.real(np.fft.ifftn(f))


def load(p):
    return np.asarray(nib.load(p).dataobj, dtype=float)


def pair_corr(vols, mask):
    h = [fine_scale(v) for v in vols]
    return [float(np.corrcoef(a[mask], b[mask])[0, 1]) for a, b in itertools.combinations(h, 2)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("maps", nargs="+")
    ap.add_argument("--scans", nargs="*", default=[])
    a = ap.parse_args()
    maps = [load(p) for p in a.maps]
    for i, v in enumerate(maps, 1):
        pk = [[(k, round(r, 1)) for k, r in top_peaks(p)] for p in axis_profiles(v)]
        print(f"map {i}: top peaks per axis (k, prominence): {pk}")
    mask = maps[0] > 0
    if len(maps) > 1:
        print("fine-scale correlation between maps of different subjects:", [round(c, 2) for c in pair_corr(maps, mask)])
    if len(a.scans) > 1:
        print("same measure for the subjects' scans (anatomical control):",
              [round(c, 2) for c in pair_corr([load(p) for p in a.scans], mask)])
    for i, p in enumerate(a.scans, 1):
        pk = [[(k, round(r, 1)) for k, r in top_peaks(q)] for q in axis_profiles(load(p))]
        print(f"scan {i}: top peaks per axis (k, prominence): {pk}")


if __name__ == "__main__":
    main()
