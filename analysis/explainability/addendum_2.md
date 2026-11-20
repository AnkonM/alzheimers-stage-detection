# Addendum 2: placement of the CNN attribution map and region table in the report

Date: 2026-10-23. Made after the results, like Addendum 1 (2026-10-05). It changes no threshold, verdict or number.

## Status of the evidence (unchanged)
The LayerCAM map of the ALZ CNN (layer1) fails both pre-registered sanity checks (cascade randomisation, region deletion). Addendum 1 withdrew every anatomical claim. This addendum only says where the map and its 12-region table may appear in the report.

## Rules
1. **Clinician view:** the skull-stripped scan, the ALZ number and the volume percentiles. No overlay, no region table, no CNN-derived text.
2. **Technical view only:** the LayerCAM overlay and the 12-region density table, under a FAIL badge for both sanity checks and the fixed wording "attribution density, not evidence of disease location".
3. **Table format:** real atlas row names, in fixed atlas order, never sorted and never colour-scaled by density. The AD−CN comparison column is always visible.
4. **Wording ban:** the string "hippocamp" appears nowhere in the report except inside that table element. This covers prose, captions, tooltips, alt text and PDF text. The DOM test enforces it.
5. **Probabilities** are labelled "model probability". The word "calibrated" is not used anywhere.

## Why
The table's row names are labels of the atlas, not findings. Naming the rows is required for the table to be readable. Colouring or sorting by density would turn it into a ranking of "where the disease is", which the failed checks do not support.

## Amendment, 2026-10-28: scope of rule 4
Rule 4 applies to **CNN attribution content only**. Inside the CNN panel of the technical view, "hippocamp" may appear only in the region table. Outside that panel, the measured hippocampal volume row and the MRI-specialist driver "Hippocampal volume / ICV" are kept. They are measured model inputs, not CNN output.

Added separation rules, enforced by `tests/test_report_panels.py`:
- Text in the CNN panel never refers to volumes, percentiles or drivers.
- Text in the volume and driver panels never refers to the overlay or the attribution map.

## Amendment, 2026-11-04: lattice check for future explanation candidates
The layer1 LayerCAM overlay shows a regular lattice with a period of 8 voxels (see `lattice_note.md`). Nothing in this addendum changes: the
overlay stays in the technical view only, with both FAIL badges and the fixed wording, and no smoothing or thresholding is applied.
Any future explanation candidate must also be checked for the same spectral peak, alongside the two pre-registered sanity checks, and
the result reported with them.
