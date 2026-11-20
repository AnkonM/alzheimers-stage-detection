# SynthSeg vs FreeSurfer 5.1: descriptive agreement (not a gate)

n = 25 of 25 sampled train scans (failures: 0). SynthSeg QC min per scan 0.67-0.77; below 0.65: 0; segmentations touching the patch boundary: 0.

| Quantity | ICC(2,1) [95% CI] | Bias, % of FS mean | 95% limits of agreement, % | Pearson r |
|---|---|---|---|---|
| Ventricles / ICV | 0.89 [0.81, 0.93] | +21.5 | +3 to +40 | 0.99 |
| Hippocampus / ICV | 0.85 [0.69, 0.91] | +8.8 | -5 to +22 | 0.94 |
| WholeBrain / ICV | 0.52 [0.16, 0.70] | +1.0 | -7 to +9 | 0.57 |
| Entorhinal / ICV | 0.15 [-0.01, 0.26] | +31.1 | -8 to +70 | 0.43 |
| Fusiform / ICV | 0.28 [-0.21, 0.66] | +2.6 | -22 to +27 | 0.29 |
| MidTemp / ICV | 0.33 [0.10, 0.54] | -11.9 | -30 to +7 | 0.56 |
| Ventricles (raw mm3) | 0.92 [0.87, 0.93] | +21.5 | +10 to +33 | 1.00 |
| Hippocampus (raw mm3) | 0.86 [0.73, 0.91] | +9.1 | -3 to +21 | 0.95 |
| WholeBrain (raw mm3) | 0.96 [0.91, 0.98] | +1.4 | -3 to +6 | 0.97 |
| Entorhinal (raw mm3) | 0.19 [0.08, 0.29] | +31.5 | -4 to +67 | 0.59 |
| Fusiform (raw mm3) | 0.63 [0.36, 0.80] | +3.0 | -16 to +22 | 0.65 |
| MidTemp (raw mm3) | 0.59 [0.39, 0.73] | -11.5 | -27 to +4 | 0.82 |
| ICV (raw mm3) | 0.85 [0.72, 0.93] | +0.2 | -10 to +11 | 0.85 |

Cortical ratios with ICC(2,1) below 0.5: Entorhinal, Fusiform, MidTemp.
