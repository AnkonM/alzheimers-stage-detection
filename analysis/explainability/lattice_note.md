# Lattice in the layer1 LayerCAM maps (diagnostic note, 2026-11-04)

A regular lattice of blobs is visible in the CNN overlay of the report's technical view. This note records what was measured. It changes no
method, threshold or verdict: the map stays in the technical view, labelled FAIL for both sanity checks, with the wording "attribution
density, not evidence of disease location" (Addendum 1 and Addendum 2). The CNN panel and the explainability element are to be reworked later.

## What was measured
Maps analysed: the overlay of each demo case, that is the layer1 LayerCAM of the ALZ logit, upsampled trilinearly from the 24-cell grid
to 96^3, masked to the template brain, unit positive mass, **averaged over the 15 fold models**. Scripts: `tests/diagnostics/lattice_check.py`
(spectrum and cross-subject correlation, numpy and nibabel only) and `tests/diagnostics/lattice_gradient_parity.py` (one model, one scan).

| Measurement | Result |
|---|---|
| Dominant spectral peak of the maps, all three axes, all four demo maps | k = 12 cycles per 96 voxels, a period of 8 voxels (two layer1 cells; layer1 has a stride of 4 voxels). Prominence 5.7 to 7.2 times the neighbouring frequencies |
| The same check on the scan itself (anatomical control) | no peak at k = 12; the strongest components are at k = 3 to 5 |
| Fine-scale component (period under 12 voxels) correlated between the maps of different subjects | 0.94 to 0.95 for all six pairs |
| The same measure on those subjects' scans | 0.23 to 0.26 |
| Mean gradient magnitude at the layer1 cells by parity of the cell index, one model and one scan | (even, even, even) cells 2.26 times the mean of the eight classes; the other seven classes 0.61 to 1.23 |

## Reading
- The lattice is fixed in grid coordinates, not tied to anatomy: it correlates across different subjects, which the scans do not.
- **It survives averaging over the 15 fold models.** The maps measured are already the 15-model average, so it is shared by the fold models
  and is not per-model noise that averaging would remove.
- The gradient-parity result is **consistent with** the stride-2 entry of layer2 sampling layer1 unevenly, which would give a period of two
  cells. It was measured on one model and one scan, so it does not establish the mechanism.

## Requirement for any future explanation candidate
Besides the two pre-registered sanity checks (cascading randomisation and region deletion), any candidate attribution map must also be
checked for the same spectral peak with `tests/diagnostics/lattice_check.py`: the per-axis spectrum and, across different subjects, the
fine-scale correlation of the maps against the scans' own. The result is reported next to the sanity checks. No pass criterion is set here;
when a new method is pre-registered, the criterion for this check is declared in that pre-registration before the maps are produced.
