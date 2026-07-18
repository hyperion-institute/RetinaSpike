# RetinaSpike

A hybrid CNN–SNN pipeline that encodes images into sparse spike trains using a **retina-inspired, contrast-based saliency mechanism**, rather than raw pixel intensity, and classifies them with a spiking neural network trained via surrogate gradients.

## Inspiration

The biological retina does not transmit raw light intensity to the brain. Retinal circuits perform local preprocessing — lateral inhibition and center-surround receptive fields — that suppresses redundant, homogeneous regions and emphasizes edges, contours, and structurally distinctive areas before the signal ever reaches downstream visual cortex. This is the same principle behind classical rank-order / latency coding models of early vision, where the most salient regions are transmitted first, and where scarce neural resources are spent on the parts of a scene that actually carry information.

Most existing spike-encoding methods for SNNs approximate this by computing contrast directly on **raw pixel intensity** — typically with a fixed Difference-of-Gaussians filter acting as a stand-in for a retinal ganglion cell's receptive field. That filter is hand-designed and task-agnostic: it responds to the same kind of intensity edge whether or not that edge is actually useful for the downstream task.

## The idea

RetinaSpike asks a different question. Instead of *"how bright is this pixel relative to its neighbors?"*, it asks *"how different is this location from its neighbors, in a representation the network has learned to be useful for the task?"*

Concretely: a CNN feature extractor first maps the image into a higher-dimensional, L2-normalized feature space. Local contrast is then computed in that space, via cosine similarity between a location's feature vector and its 3×3 neighborhood — low similarity (i.e. a structurally distinctive location) produces a strong contrast response, uniform neighborhoods are suppressed. That contrast map is converted into a spike train by latency coding: high-contrast locations fire early, low-contrast locations fire late or not at all.

The hypothesis this project is testing: because the contrast is computed on a *learned* representation rather than a fixed pixel-level filter, the notion of "structurally distinctive" can in principle be shaped by what's actually relevant to the classification task, rather than being fixed to generic intensity edges. Whether the current training setup actually exploits this — as opposed to producing a fixed filter in disguise — is discussed honestly below, since it turns out to be the crux of the whole approach.

## Pipeline

```
Image → CNN feature extractor → L2-normalized features
      → local cosine-similarity contrast (3×3 neighborhood)
      → latency spike encoding (high contrast → early spike, low contrast → no spike)
      → spiking CNN classifier (LIF neurons, surrogate gradient)
      → class prediction
```

Full architectural details, the complete derivation, and the biological motivation in depth are in [`RetinaSpike_report.pdf`](./RetinaSpike_report.pdf).

## Results (MNIST, 6 seeds)

| Timestep (T) | Threshold | Mean accuracy | Internal SNN spikes / img |
|---|---|---|---|
| 4  | 0.0 | 98.23% ± 0.15 | ~1,040 |
| 8  | 0.0 | 98.20% ± 0.20 | ~1,980 |
| 12 | 0.0 | 98.26% ± 0.11 | ~2,810 |

At `threshold = 0`, accuracy is statistically indistinguishable across T = 4, 8, 12 (differences within 1 standard deviation), while internal spike activity scales roughly linearly with T. In other words, **T can be reduced from 12 to 4 with no measurable accuracy loss, at roughly a 2.7× reduction in spike activity** — the main efficiency finding of this project so far.

Raising the sparsity threshold trades accuracy for sparsity smoothly up to `threshold = 0.6`, and degrades sharply — and less reproducibly across seeds — at `threshold = 0.8`. Full sweep results across all T × threshold × seed combinations are in the report.

## Known limitations

- The spike-selection step (`contrast > threshold`) is a hard, non-differentiable cutoff. Gradients currently reach the feature extractor only through the sub-timestep latency placement, not through the decision of *which* locations spike — meaning the central hypothesis above (a learned, task-shaped notion of contrast) is only partially tested by the current training setup. This is the most important open problem in the project, and is discussed in detail in the report.
- Evaluated on MNIST only; no comparison yet against a pixel-level (DoG-based) contrast baseline or standard rate coding, so it isn't yet established that learned-feature contrast outperforms the classical fixed-filter approach.
- Single dataset, no energy or hardware measurements — sparsity is reported here as a proxy for efficiency, not a validated one.

## Related work

This project builds on a long line of retina- and contrast-inspired latency coding for SNNs — most notably DoG-filter contrast combined with rank-order coding on raw pixels. The distinction explored here is applying the same contrast → latency recipe to a *learned* feature space rather than pixel intensity. See the report for a full discussion and citations.

## Where this is going

Open questions the project is currently working through: whether a differentiable (soft) spike-selection mechanism changes the learned contrast in a meaningful way, whether the learned-feature approach actually beats a fixed DoG-filter baseline under matched conditions, and whether the efficiency finding above (T-invariance of accuracy) holds on harder datasets than MNIST. Longer-term directions include adaptive contrast thresholds, multi-scale receptive fields, and real event-camera data, where the retina-inspired framing would correspond to an actual sensor rather than a simulated one.
