# BdSL-NMM Canonical Results

## Evaluation Protocol
All results use Leave-One-Signer-Out (LOSO) evaluation.
Signer S7 is the held-out test signer (601 samples).
Training uses signers S1-S6 (3,214 samples).

## Word Recognition Results (62 classes, LOSO)

| Model                | Top-1  | Top-5  |
|----------------------|--------|--------|
| Random chance        | 1.61%  | 8.06%  |
| SingleStreamBaseline | 43.59% | 82.03% |
| BdSL-SPOTER (reprod.)| 56.41% | 89.02% |
| SignNetV2 word-only  | 43.26% | 82.03% |
| SignNetV2 multitask  | 38.44% | 78.20% |

## Expression Recognition (SignNetV2 multitask, 5 classes)

| Class    | Accuracy |
|----------|----------|
| Neutral  | 29.08%   |
| Happy    | 0.00%    |
| Sad      | 9.17%    |
| Negation | 0.00%    |
| Question | 60.50%   |
| Overall  | 20.63%   |
| Random   | 20.00%   |

## Training Times
GPU (RTX 4080 Super): ~30-40 minutes per run
CPU: ~8-12 hours per run

## Notes
- BdSL-SPOTER originally reported 97.92% on BdSLW60
  under random split. Under our strict LOSO protocol
  it achieves 56.41%.
- SignNetV2 word-only (43.26%) matches baseline (43.59%)
  confirming the architecture works without multitask cost.
- SignNetV2 multitask (38.44%) adds expression recognition.
  The 4.82pp gap is the measurable cost of simultaneously
  classifying 5 expression classes.
