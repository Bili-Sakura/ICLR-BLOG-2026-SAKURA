---
layout: post
title: "Tobler's Law in the Age of Transformers: A Geographic Perspective on Neural Architectures"
date: 2026-04-27
categories: blog
tags: [deep-learning, geography, transformers, cnns, gnns]
---

> "Everything is related to everything else, but near things are more related than distant things." — Waldo Tobler, 1970 {% cite tobler1970computer %}

Fifty-six years ago, Waldo Tobler articulated the First Law of Geography. While originally intended to describe demographic growth and urban spreading in the Detroit region, this simple axiom has implicitly governed the design of neural networks for decades.

In the field of Deep Learning, we often frame progress as a battle between inductive biases and the "Bitter Lesson" of scale {% cite richsuttonBitterLesson2019 %}. However, a revisiting of recent literature suggests a different narrative: the history of modern neural architectures—from CNNs to Transformers and beyond—is essentially a debate about the definition of "nearness."

In this post, we re-interpret the evolution of state-of-the-art architectures not as a rejection of Tobler's Law, but as a continuous refinement of the metric space in which it applies.

### The Euclidean Orthodoxy: Convolutional Neural Networks

For a long time, "nearness" was literal. In the era dominated by LeNet-5 {% cite lecunGradientbasedLearningApplied1998 %} and its successors, the topology of the data was assumed to be a fixed Euclidean grid (images). Convolutional Neural Networks (CNNs) enforced Tobler's Law via **inductive bias**: hard-coded local kernels assumed that a pixel is most related to its immediate neighbors.

This approach was incredibly efficient. By sharing weights across the spatial dimension, CNNs respected translational equivariance—a geometric symmetry. However, this rigid adherence to Euclidean distance created a "horizon" of context. To relate distant things, one needed to stack layers, slowly expanding the receptive field.

### Temporal Neighborhoods: Recurrent Networks

Sequence models such as LSTMs respected a *temporal* version of Tobler's Law: the gating equations couple each hidden state most strongly with its immediate past, while the memory cell lets only a handful of distant events remain "near" enough to matter {% cite hochreiterLongShortTermMemory1997 %}. Even before Transformers, we were already redefining distance—not in pixels, but in timesteps—and carefully engineering pathways (forget/input gates) that modulate which relationships remain local and which are promoted to global context.

### The Death of Distance: The Transformer Revolution

The introduction of the Transformer {% cite vaswaniAttentionAllYou2017 %} seemed to mark the death of geography. The self-attention mechanism allowed any token to attend to any other token, regardless of distance. In the Transformer's view, "everything is related to everything else," full stop. The distance penalty was removed.

Or was it?

While the architecture allowed for global connectivity, it immediately became apparent that ignoring "nearness" entirely was suboptimal. The authors had to inject Positional Encodings to remind the model that the order of the sequence mattered.

### The Return of Geography: Relative Positions and Convolutions

Recent research indicates that the "death of distance" was exaggerated. We are witnessing a synthesis where global attention is being enriched by local geographic priors.

**1. Self-Attention is a Convolution (if you squint)**
Research by Cordonnier et al. {% cite cordonnierRelationshipSelfAttentionConvolutional2020 %} and Chang et al. {% cite changConvolutionsSelfAttentionReinterpreting2021 %} demonstrated that multi-head self-attention often learns to behave like a convolution. Chang et al. showed that relative position embeddings—effectively defining "nearness" by token distance rather than absolute index—are mathematically equivalent to dynamic lightweight convolutions. They proposed *Composite Attention*, uniting the global reach of attention with the local priors of convolution.

**2. Restoring Translational Equivariance**
Standard Transformers lack translational equivariance, a key property for robustness in vision. Horn et al. {% cite hornTranslationalEquivarianceKernelizable2021 %} introduced Relative Positional Encodings into kernelizable attention mechanisms (Performers), proving that re-injecting this "geographic" bias improves robustness to shifts in input data.

**3. Unifying the Paradigms**
The paper *Translution* {% cite fanTranslutionUnifyingSelfattention2025 %} takes this a step further. It argues that modeling data involves two steps: identifying relevant elements (Attention) and encoding them (Convolution). By proposing an operation that unifies adaptive identification with relative encoding, they outperform pure self-attention. This suggests that the optimal architecture is one that can dynamically modulate between the "global village" of attention and the "local neighborhood" of convolution.

### Pyramids and Skip-Connections: Learning Multi-Scale Neighborhoods

Architectures such as U-Net implement Tobler's Law across multiple spatial scales: encoder stages diffuse information outward, while decoder skip-connections re-inject fine-grained locality so that distant context never erases nearby detail {% cite taiMathematicalExplanationUNet2024 %}. Stand-alone vision Transformers reinforce the same principle computationally by constraining early layers to windowed attention before gradually relaxing the radius {% cite ramachandranStandAloneSelfAttentionVision2019 %}. In both cases, the model is explicitly taught how "nearness" should evolve as one zooms out.

### The Manifold Hypothesis: Transformers as GNNs

Perhaps the most striking re-interpretation comes from viewing Transformers through the lens of Graph Neural Networks (GNNs). Joshi {% cite joshiTransformersAreGraph2025 %} argues that Transformers are simply GNNs operating on a fully connected graph.

In this view, Tobler's Law still holds, but the "space" is no longer the 2D grid of pixels or the 1D line of text. The space is the **latent manifold** learned by the model. Attention weights define the topology: if two tokens have a high attention score, they are "near" in the semantic space, regardless of their position in the input sequence.

Graph Attention Networks make this analogy explicit: each node computes a learned kernel over its immediate neighbors, enforcing a sparse and anisotropic neighborhood structure {% cite velickovicGraphAttentionNetworks2018 %}. The attention coefficients are therefore nothing more than Toblerian weights on an irregular manifold—generalizing urban neighborhoods to molecular graphs, program graphs, or knowledge graphs.

### Oversmoothing: When Tobler's Law Goes Too Far

Tobler himself noted in his 1970 simulation that his model introduced an "excessive amount of smoothing," converging towards a simplified mean rather than capturing the sharp disparities of reality {% cite tobler1970computer %}. Surprisingly, modern Transformers suffer from the exact same pathology in the semantic space.

Choi et al. {% cite choiGraphConvolutionsEnrich2024 %} identify "oversmoothing" in deep Transformers, where token representations across layers converge to indistinguishable values—an entropy collapse. Just as Tobler's urban model blurred distinct neighborhoods, deep self-attention can blur distinct concepts. Their solution? Re-introducing **Graph Convolutions** to enrich self-attention. By acting as high-pass filters, these convolutions sharpen the "geographic" features of the latent manifold, preventing the collapse of distance.

### The Semantic Tobler's Law: System 2 Inductive Biases

If we accept that "nearness" is semantic, how do we define the metric? Goyal and Bengio {% cite goyalInductiveBiasesDeep2022a %} offer a compelling answer through the lens of "System 2" deep learning. They argue for inductive biases that enforce a **sparse factor graph** in the space of high-level variables.

This is the *Semantic Tobler's Law*: concepts are not related to *all* other concepts, but only to a sparse few that are causally "near."

* **System 1 (Perception)**: Dense, parallel, global. Everything is related to everything.
* **System 2 (Reasoning)**: Sparse, sequential, local. Cause A is related only to Effect B.

By enforcing sparsity in the dependency graph of high-level variables, we are effectively re-imposing a geographic constraint on the "mind" of the neural network.

### The Hardware Lottery: Why Density Won (For Now)

If locality and sparsity (the essence of Tobler's Law) are so computationally efficient and biologically plausible, why did the dense, quadratic Transformer win?

Sara Hooker's concept of the *Hardware Lottery* {% cite hookerHardwareLottery2021 %} provides the answer. We did not choose Transformers solely because they were theoretically superior; we chose them because our hardware (GPUs/TPUs) excels at dense matrix multiplications. Sparse, local operations (like those in GNNs or biological brains) are often less efficient on modern accelerators than brute-force dense compute.

However, the "Bitter Lesson" {% cite richsuttonBitterLesson2019 %} suggests that general-purpose methods that leverage computation scale best. Yet, as we reach the limits of dense scaling, the pendulum swings back. The work of Fan et al. (Translution) and Horn et al. (Translational Equivariance) suggests that we are re-discovering the efficiency of "nearness"—not as a hard constraint, but as a learned or soft prior to break the bottleneck of quadratic complexity.

### Conclusion

Modern neural networks have not disproven Tobler's Law; they have generalized it.

1. **CNNs** applied Tobler's Law in **Euclidean Space** (Hard Inductive Bias).
2. **Transformers** apply Tobler's Law in **Latent Semantic Space** (Soft/Learned Bias).
3. **Modern Hybrid Architectures** (Translution, Graph Transformers) acknowledge that while global context is powerful, the structural priors of locality are necessary for data efficiency, robustness, and avoiding oversmoothing.

As we look toward the next generation of architectures, we see a trend not of discarding the past, but of unifying the global context of attention with the local structure of convolution. We are building machines that learn their own geography.

---
