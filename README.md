# RetinaSpike: Bio-Inspired Contrast-Encoded Spiking Neural Network

RetinaSpike is a research and implementation project of a hybrid **CNN-SNN** architecture that accurately mimics the primary visual processing mechanism of the **Human Retina**. Instead of feeding raw pixel data directly into deep neural layers, the system utilizes a feature extractor combined with a Cosine Similarity-based **Lateral Inhibition** mechanism to convert images into sparse, information-rich, and energy-efficient spiking sequences.

---

## 👁️ Bio-Inspired Mechanism: From Retina to Spike Encoding

In natural visual systems, the retina does not act like a conventional camera that transmits raw pixels directly to the brain. Instead, it functions as a highly sophisticated data pre-processor. 

The architecture of **RetinaSpike** is specifically designed to replicate this 3-tier biological structure:
### 1. Primary Visual Cortex & Ganglion Cells (Feature Extractor)
* **Biological Counterpart:** Photoreceptors receive light and pass signals through ganglion cells to extract low-level features such as orientation, edges, and corners.
* **In-Code Implementation:** The `FeatureExtractor` class (consisting of Conv2d + ReLU layers) acts as biological receptive fields. It transforms a 1-channel raw grayscale image into 64 channels of abstract latent features, followed by L2 normalization to eliminate intensity fluctuations and illumination noise.

### 2. Lateral Inhibition via Horizontal Cells (Cosine Similarity)
* **Biological Counterpart:** Horizontal cells implement a mechanism known as **Lateral Inhibition**. When a specific neuron fires, it inhibits its neighboring neurons. This mechanism makes the human eye exceptionally sensitive to **contrast, boundaries, and sudden structural transitions**, while effectively filtering out homogenous, redundant backgrounds.
* **In-Code Implementation:** The `compute_contrast` function models this exact phenomenon. By sliding a $3 \times 3$ window across the feature maps, the system computes the **Cosine Similarity** between the center pixel and its 8 neighbors.
  * If the neighborhood shares identical features with the center (homogenous region), the similarity is high $\rightarrow$ Contrast drops to $0$ (Fully Inhibited).
  * If the neighborhood differs significantly (structural edge), the similarity is low $\rightarrow$ Contrast spikes up (Highly Excited).

### 3. Time-to-First-Spike / Latency Coding
* **Biological Counterpart:** Stimuli with higher contrast cause optic nerve cells to accumulate membrane potential faster, prompting them to **fire spikes earlier** to transmit urgent information to the brain.
* **In-Code Implementation:** The `encode_to_spike` function filters out regions where contrast exceeds a predefined `THRESHOLD`. Vips of maximum contrast are mapped to fire at the earliest timesteps ($t \rightarrow 0$), while lower contrast regions fire later, creating a binary spike tensor sequence over time with the shape `[T, B, 1, H, W]`.

---

## ⚡ Data Pipeline Architecture

The end-to-end data processing workflow operates as follows:

1. **Input Data:** Raw MNIST images $[B, 1, 28, 28]$ are fed into the network.
2. **Feature Extraction:** Maps the spatial input into a latent feature space of $[B, 64, 7, 7]$ and enforces unit vector normalization.
3. **Contrast Computing:** Evaluates local Cosine distances to yield a distinctive contrast map $[B, 7, 7]$.
4. **Latency Encoding:** Translates the contrast map into real-time binary spike sequences of size $[T, B, 1, 7, 7]$.
5. **SNN Classification:** The temporal spike streams pass through a Spiking Neural Network consisting of Convolutional and **Leaky Integrate-and-Fire (LIF)** layers, accumulating membrane potentials to output the final classification.

---

## 🛠️ Configuration & Core Hyperparameters

The model is optimized for hardware-friendly deployment, prioritizing high network sparsity:

* **TIMESTEP  = 12**: The number of simulation time steps for spiking neurons.
* **THRESHOLD = 0.4**: The quantile threshold for contrast filtering (effectively zeroing out background noise).
* **Surrogate Gradient**: Employs `fast_sigmoid` to bypass the non-differentiable nature of discrete binary spikes during backpropagation.

---

## 📈 Key Advantages

* **High System Sparsity:** Driven by retinal lateral inhibition, only critical information triggers spikes. This drastically cuts down redundant operations, paving the way for low-power deployment on neuromorphic hardware chips.
* **Lossless Spatial Encoding:** Unlike time-consuming rate coding methods, contrast-latency coding preserves the structural and spatial correlations of objects within a very short simulation window.
