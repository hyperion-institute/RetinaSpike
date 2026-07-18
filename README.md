# RetinaSpike

A hybrid CNN–SNN pipeline that encodes images into sparse spike trains using a **retina-inspired, contrast-based saliency mechanism**, rather than raw pixel intensity, and classifies them with a spiking neural network trained via surrogate gradients.

## Inspiration

The biological retina does not transmit raw light intensity to the brain. Retinal circuits perform local preprocessing — lateral inhibition and center-surround receptive fields — that suppresses redundant, homogeneous regions and emphasizes edges, contours, and structurally distinctive areas before the signal ever reaches downstream visual cortex. This is the same principle behind classical rank-order / latency coding models of early vision, where the most salient regions are transmitted first.

RetinaSpike follows this idea, with one deliberate departure from most prior contrast-encoding SNN work: instead of computing contrast on **raw pixel intensity** (e.g. via Difference-of-Gaussians filters), it computes contrast on a **learned CNN feature representation**, using cosine similarity between a location's feature vector and its 3×3 neighborhood. The hypothesis: a learned feature space may produce a more task-relevant notion of "structurally distinctive" than a fixed pixel-level filter.

## Pipeline

```
Image → CNN feature extractor → L2-normalized features
      → local cosine-similarity contrast (3×3 neighborhood)
      → latency spike encoding (high contrast → early spike, low contrast → no spike)
      → spiking CNN classifier (LIF neurons, surrogate gradient)
      → class prediction
```

Full architectural details, the biological motivation, and the complete derivation are in [`RetinaSpike_report.pdf`](./RetinaSpike_report.pdf).

## Results (MNIST, 6 seeds)

| Timestep (T) | Threshold | Mean accuracy | Internal SNN spikes / img |
|---|---|---|---|
| 4  | 0.0 | 98.23% ± 0.15 | ~1,040 |
| 8  | 0.0 | 98.20% ± 0.20 | ~1,980 |
| 12 | 0.0 | 98.26% ± 0.11 | ~2,810 |

At `threshold = 0`, accuracy is statistically indistinguishable across T = 4, 8, 12 (differences within 1 standard deviation), while internal spike activity scales roughly linearly with T. In other words, **T can be reduced from 12 to 4 with no measurable accuracy loss, at roughly a 2.7× reduction in spike activity** — the main efficiency finding of this project so far.

Raising the sparsity threshold trades accuracy for sparsity smoothly up to `threshold = 0.6`, and degrades sharply (and less reproducibly across seeds) at `threshold = 0.8`. Full sweep results (all T × threshold × seed combinations) are in the report.

## Known limitations

- The spike-selection step (`contrast > threshold`) is a hard, non-differentiable cutoff. Gradients currently reach the feature extractor only through the sub-timestep latency placement, not through the decision of *which* locations spike — meaning the learned-feature-contrast hypothesis is only partially tested by the current training setup. Documented in detail in the report.
- Evaluated on MNIST only; no comparison yet against a pixel-level (DoG-based) contrast baseline or standard rate coding.
- Single dataset, no energy/hardware measurements — sparsity is reported as a proxy, not a validated efficiency metric.

## Installation

```bash
pip install torch torchvision snntorch
```

## Usage

```bash
python train.py
```

Trains `FeatureExtractor` + `SNNModel` jointly for `EPOCHS` epochs, evaluates on the MNIST test set each epoch, and saves the best checkpoint to `RetinaSpike.pth`. Key hyperparameters (`TIMESTEP`, `THRESHOLD`, `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`) are set at the top of the script.


## Related work

This project builds on a long line of retina-/contrast-inspired latency coding for SNNs (e.g. DoG-filter contrast + rank-order coding on raw pixels). The distinction explored here is applying the same contrast → latency recipe to a *learned* feature space rather than pixel intensity. See the report for a full discussion and citations.
