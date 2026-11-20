# LayerCAM region densities: engine vs the old card (known difference)

Measured 2026-11-03 for one ADNI demo case with `tests/diagnostics/cam_density_batch_diag.py` (frozen CNNs, `src/card/build_cam.py` arithmetic).

| Computation | max abs difference to the stored card density |
|---|---|
| card batching (chunk of 32 subjects, CAM batch size 8), card arithmetic | 5.6e-8 |
| single scan (batch size 1, how the engine runs) | 1.34e-3 |

Cause: GPU batch composition. The convolutions give slightly different floating-point results for the same scan depending on which scans share a batch, as for ALZ (`PARITY_SPEC.md`, amendment 2). It is not a code difference: scoring the scan in the card's batching reproduces the card to 5.6e-8.
Status: a known difference, not gated. Densities of a single-scan run can differ from the old card's by about 1e-3 (relative to densities near 1). No tolerance was set to hide it.
