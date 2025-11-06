---
layout: distill
title: "From U-Nets to DiTs: The Architectural Evolution of Text-to-Image Diffusion Models (2021–2025)"
description: "A comprehensive analysis of how diffusion model architectures evolved from U-Net backbones to Diffusion Transformers, transforming text-to-image generation capabilities."
date: 2026-04-27
future: true
htmlwidgets: true

# Anonymize when submitting
authors:
  - name: Anonymous

# must be the exact same name as your blogpost bibliography file
bibliography: 2026-04-27-diffusion-architecture-evolution.bib

# Table of contents: names must exactly match section headings below
toc:
  - name: "TL;DR"
  - name: "Preliminaries: Diffusion Models for Image Generation"
  - name: "Interactive Timeline"
  - name: "The U-Net Era"
  - name: "The DiTs Era"
  - name: "Pre-trained Text-to-Image Checkpoints"
  - name: "Experiments and Case Studies"
  - name: "Discussion"
---

---

## TL;DR

- **What changed**: Architectural evolution from U-Net to DiT (Diffusion Transformer) backbones fundamentally transformed diffusion-based text-to-image generation capabilities from 2021–2025.
- **Core insight**: While U-Net models hit scaling bottlenecks, DiTs leverage transformer architectures to achieve better scalability and generation performance.
- **Goal**: Provide a comprehensive analysis of architectural evolution and its implications for text-to-image generation.
- **Key finding**: The shift to DiTs enabled unprecedented scaling, improved text understanding, and opened new possibilities for multimodal integration.

## Preliminaries: Diffusion Models for Image Generation

Diffusion models have emerged as a powerful paradigm for generative modeling by learning to reverse a gradual noise corruption process. The fundamental approach involves two key stages: a **forward diffusion process** that systematically adds noise to data until it becomes pure Gaussian noise, and a **reverse denoising process** where a neural network gradually removes this noise to generate new samples.

This framework has demonstrated remarkable success across diverse domains including image generation, audio synthesis, video generation, and even applications in natural language processing and molecular design. The generality of the diffusion framework makes it particularly attractive for complex generative tasks.

{% include figure.liquid path="assets/img/2026-04-27-diffusion-architecture-evolution/lilian_DDPM.png" alt="Diagram showing the forward noising process and the reverse denoising process in diffusion models." caption='The Markov chain for the forward and reverse diffusion processes, which generate a sample by slowly adding (and removing) noise. Image Credit: <d-cite key="wengWhatAreDiffusion2021"></d-cite>' %}

For readers seeking a comprehensive introduction to diffusion model fundamentals, we recommend Yang Song's excellent exposition on [score-based generative modeling](https://yang-song.net/blog/2021/score/) <d-cite key="song2019generative"></d-cite> and Lilian Weng's detailed overview of [diffusion models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) <d-cite key="wengWhatAreDiffusion2021"></d-cite>.

## Interactive Timeline

<div class="l-page">
  <iframe src="{{ 'assets/html/2026-04-27-diffusion-architecture-evolution/timeline.html' | relative_url }}" frameborder='0' scrolling='yes' height="700px" width="100%"></iframe>
</div>

## The U-Net Era

The early pioneering works in diffusion-based image generation predominantly adopted **U-Net architectures** <d-cite key="ronnebergerUNetConvolutionalNetworks2015"></d-cite> as their neural network backbone. This choice was largely influenced by U-Net's proven success in various computer vision tasks <d-cite key="linRefineNetMultiPathRefinement2017"></d-cite><d-cite key="salimansPixelCNNImprovingPixelCNN2017"></d-cite>.

The foundational models in this era established the core principles of diffusion-based generation. **NCSN** (Noise Conditional Score Network) <d-cite key="song2019generative"></d-cite> pioneered score-based generative modeling using a RefineNet backbone <d-cite key="linRefineNetMultiPathRefinement2017"></d-cite>, while **DDPM** (Denoising Diffusion Probabilistic Models) <d-cite key="hoDenoisingDiffusionProbabilistic2020"></d-cite> established the probabilistic framework using a PixelCNN++ architecture <d-cite key="salimansPixelCNNImprovingPixelCNN2017"></d-cite>. Subsequent refinements including **NCSNv2** <d-cite key="songImprovedTechniquesTraining2020"></d-cite>, **IDDPM** <d-cite key="nicholImprovedDenoisingDiffusion2021"></d-cite>, **ADM** (Ablated Diffusion Model) <d-cite key="dhariwalDiffusionModelsBeat2021"></d-cite>, and **SDE** (Score-based Diffusion via Stochastic Differential Equations) <d-cite key="songScoreBasedGenerativeModeling2021"></d-cite> built upon these foundations with architectural variations similar to DDPM or NCSN. However, these early models focused primarily on unconditional image generation and lacked text-to-image capabilities.

The breakthrough for text-to-image generation came with **LDM** (Latent Diffusion Models, also known as Stable Diffusion) <d-cite key="rombachHighResolutionImageSynthesis2022"></d-cite>, which introduced a latent U-Net architecture combined with a KL-regularized VAE (autoencoder) to enable efficient text-conditioned generation. Following this success, several notable U-Net-based text-to-image models emerged, each exploring different architectural innovations within the U-Net paradigm:

| Model                                                                                                                           | Gen. (#Param) | Txt. (#Param) | Total (#Param) | Release Date |
| ------------------------------------------------------------------------------------------------------------------------------- | ------------- | ------------- | -------------- | ------------ |
| SD v2.1 <d-cite key="rombachHighResolutionImageSynthesis2022"></d-cite>                                                         | 0.87B         | 0.34B         | 1.29B          | 2022-12-07   |
| Kandinsky <d-cite key="razzhigaevKandinskyImprovedTexttoImage2023"></d-cite>                                                    | TBD           | TBD           | TBD            | 2023-01-01   |
| UniDiffuser <d-cite key="baoOneTransformerFits2023"></d-cite>                                                                   | 0.95B         | 0.21B         | 1.25B          | 2023-05-12   |
| SDXL <d-cite key="podellSDXLImprovingLatent2024"></d-cite>                                                                      | 2.57B         | 0.82B         | 3.47B          | 2023-06-25   |
| Kandinsky 3 <d-cite key="arkhipkinKandinsky30Technical2024"></d-cite><d-cite key="arkhipkinKandinsky3TexttoImage2024"></d-cite> | TBD           | TBD           | TBD            | 2023-12-11   |
| Stable Cascade (Würstchen) <d-cite key="perniasWurstchenEfficientArchitecture2024"></d-cite>                                    | 1.56B         | 0.69B         | 2.28B          | 2024-02-07   |

The standard U-Net architecture for diffusion models typically consists of an **encoder** that progressively downsamples the noisy input, a **bottleneck** middle block that processes compressed representations, and a **decoder** that upsamples back to the original resolution. Crucially, **skip connections** preserve fine-grained spatial information across corresponding encoder and decoder stages.

{% include figure.liquid path="assets/img/2026-04-27-diffusion-architecture-evolution/diffusion_unet_illustration.png" alt="U-Net backbone used in diffusion models with time conditioning injected into residual blocks and skip connections between encoder and decoder." caption='A typical U-Net backbone used in diffusion models with time conditioning. Time representation uses sinusoidal positional embeddings or random Fourier features; these time features are injected into residual blocks via simple spatial addition or adaptive group normalization layers. Image Credit: <d-cite key="CVPR2023Tutorial"></d-cite>.' %}

### Non-text conditioning: two-stage cascades and U‑ViT

Beyond text conditioning, U-Net backbones evolved substantially in non-text settings (unconditional or class-conditional), focusing on sample quality, stability, and compute efficiency:

- **Two-stage/cascaded U-Nets**: Decompose generation into a low-resolution base diffusion model and one or more super-resolution diffusion upsamplers. The base model captures global structure; specialized upsamplers (e.g., SR3) iteratively refine detail at higher resolutions. This cascade improves fidelity and stability on ImageNet and face datasets while keeping training tractable <d-cite key="hoCascadedDiffusionModels2022"></d-cite><d-cite key="sahariaImageSuperResolution2022"></d-cite>.
- **U‑ViT (ViT backbone in a U‑shaped design)**: Replace CNN residual blocks with Vision Transformer blocks while retaining long-range skip connections. U‑ViT tokenizes noisy image patches, timesteps, and (optionally) class tokens, enabling stronger global context modeling than CNN U‑Nets and achieving competitive ImageNet FID with comparable compute <d-cite key="baoAllAreWorthWords2023"></d-cite>.

> **Key takeaways (non-text U-Net family)**
>
> - Cascades separate global structure (base) from high-frequency detail (super‑res), scaling quality to high resolutions efficiently.
> - ViT backbones in U‑shaped layouts preserve inductive benefits of skip connections while capturing long-range dependencies.
> - These ideas later influenced text-to-image systems (e.g., two‑stage SDXL) even as the field transitioned toward DiT backbones.
>   {: .block-warning}

## The DiTs Era

As U-Net–based models began to hit a scaling ceiling (e.g., SDXL with ~2.6B parameters <d-cite key="podellSDXLImprovingLatent2024"></d-cite>), naive scaling proved ineffective, motivating a shift towards alternative backbones. The introduction of Diffusion Transformers (DiTs) <d-cite key="peeblesDiTScalableDiffusion2023"></d-cite> marks a significant paradigm shift by recasting image generation as a patch-sequence modeling problem solved with transformer blocks. This approach offers several key advantages over U-Nets, including superior **scalability** via stacked DiT blocks, the ability to capture **global context** via self-attention for long-range dependencies, and a **unified** architecture that leverages advances in multimodal integration.

{% include figure.liquid path="assets/img/2026-04-27-diffusion-architecture-evolution/dit.png" alt="DiT Architecture." caption='The Diffusion Transformer (DiT) architecture. Left: We train conditional latent DiT models. The input latent is decomposed into patches and processed by several DiT blocks. Right: Details of our DiT blocks. We experiment with variants of standard transformer blocks that incorporate conditioning via adaptive layer norm, cross-attention and extra input tokens. Adaptive layer norm works best. Image Credit: <d-cite key="peeblesDiTScalableDiffusion2023"></d-cite>.' %}

### Non-text conditioning: DiT → SiT → LiT and architectural innovations

While many recent DiT variants target text-to-image via cross-attention, a parallel line of work advances non-text conditioning (e.g., class labels) where conditioning is injected without text encoders:

- **DiT (Diffusion Transformer)**: Patchify VAE latents and process them with Transformer blocks; inject timestep and class-label signals through adaptive layer norm (adaLN/FiLM) or extra tokens—no text cross-attention needed. DiT set strong class-conditional ImageNet baselines and scales predictably with depth/width <d-cite key="peeblesDiTScalableDiffusion2023"></d-cite>.
- **SiT (Scalable Interpolant Transformer)**: Keeps the DiT backbone but replaces the diffusion objective with an interpolant-based transport framework, improving stability and sample quality across sizes on ImageNet-256, with either deterministic or stochastic samplers. Same parameter/GFLOP budget as DiT, better FID. See the SiT project page: [scalable-interpolant.github.io](https://scalable-interpolant.github.io/).
- **LiT (Linear Diffusion Transformer)**: Replaces quadratic softmax attention with linear attention to achieve O(n) memory/time in sequence length, enabling higher resolutions and faster throughput. Practical recipes include weight inheritance from a pretrained DiT and hybrid distillation. Popularized in high-res efficient systems like SANA <d-cite key="xieSANAEfficientHighResolution2025"></d-cite> (see also [arXiv](https://arxiv.org/abs/2501.12976)).

**Training efficiency innovations** have further accelerated DiT development:

- **MDT/MDTv2 (Masked Diffusion Transformer)** <d-cite key="gaoMaskedDiffusionTransformer2023"></d-cite><d-cite key="gaoMDTv2MaskedDiffusion2024"></d-cite>: Introduces mask latent modeling where tokens are masked during training and an asymmetric encoder-decoder predicts masked tokens from unmasked ones. This enhances contextual relation learning among image parts, achieving SOTA FID of 1.58 on ImageNet with over 10× faster learning than DiT.
- **MaskDiT** <d-cite key="zhengFastTrainingDiffusion2023"></d-cite>: Masks ~50% of patches during training with an asymmetric transformer (encoder on unmasked patches, lightweight decoder on full patches) plus an auxiliary reconstruction task. Achieves competitive performance with only ~30% of DiT's training time on ImageNet 256×256 and 512×512.
- **SD-DiT (Self-supervised Discrimination DiT)** <d-cite key="zhuSDDiTUnleashingPower2024"></d-cite>: Incorporates self-supervised discriminative learning to improve DiT training efficiency and generation quality through auxiliary discrimination tasks.

**Architecture refinements** explore encoder-decoder separation and representation learning:

- **DDT (Decoupled Diffusion Transformer)** <d-cite key="wangDDTDecoupledDiffusion2025"></d-cite>: Decouples semantic encoding from high-frequency decoding with a dedicated condition encoder and specialized velocity decoder, resolving the optimization tension in standard DiTs. Achieves 1.31 FID on ImageNet 256×256 (4× faster convergence than previous DiTs) and 1.28 FID on 512×512.
- **RAE (Representation Autoencoders)** <d-cite key="zhengDiffusionTransformersRepresentation2025"></d-cite>: Replaces standard VAE with pretrained representation encoders (DINO, SigLIP, MAE) paired with trained decoders, providing semantically rich, high-dimensional latent spaces. Achieves 1.51 FID at 256×256 (no guidance) and 1.13 at both 256×256 and 512×512 (with guidance).

> **Key takeaways for non-text conditional DiTs**
>
> - **Conditioning path**: timestep + class (or other non-text signals) via adaLN/FiLM or tokens; no text cross-attention.
> - **Training efficiency**: Masked training (MDT, MaskDiT) reduces training time by 3–10× while maintaining or improving quality.
> - **Architecture evolution**: Decoupled designs (DDT) and representation autoencoders (RAE) push FID below 1.5 on ImageNet.
> - **Progression**: objective improvements (SiT) → efficiency at scale (LiT) → faster training (MDT, MaskDiT) → architectural refinement (DDT, RAE).
>   {: .block-warning}

| Model                                                                        | Gen. (#Param) | Txt. (#Param) | Total (#Param) | Release Date |
| ---------------------------------------------------------------------------- | ------------- | ------------- | -------------- | ------------ |
| PixArt-$\alpha$ <d-cite key="chenPixArtaFastTraining2024"></d-cite>          | 0.61B         | 4.76B         | 5.46B          | 2023/10/06   |
| Lumina-T2I <d-cite key="gaoLuminaT2XScalableFlowbased2025a"></d-cite>        | TBD           | TBD           | TBD            | 2024/04/01   |
| PixArt-$\Sigma$ <d-cite key="chenPIXARTSWeaktoStrongTraining2024a"></d-cite> | 0.61B         | 4.76B         | 5.46B          | 2024/04/11   |
| Lumina-Next-T2I <d-cite key="zhuoLuminaNextMakingLuminaT2X2024a"></d-cite>   | 1.75B         | 2.51B         | 4.34B          | 2024/05/12   |
| Stable Diffusion 3 <d-cite key="esserScalingRectifiedFlow2024"></d-cite>     | 2.03B         | 5.58B         | 7.69B          | 2024/06/12   |
| Flux.1-Dev <d-cite key="blackforestlabsFLUX1"></d-cite>                      | 11.90B        | 4.88B         | 16.87B         | 2024/08/02   |
| CogView3-Plus <d-cite key="zhengCogView3FinerFaster2024a"></d-cite>          | 2.85B         | 4.76B         | 8.02B          | 2024/10/13   |
| Hunyuan-DiT <d-cite key="liHunyuanDiTPowerfulMultiResolution2024a"></d-cite> | 1.50B         | 2.02B         | 3.61B          | 2024/12/01   |
| SANA <d-cite key="xieSANAEfficientHighResolution2025"></d-cite>              | 0.59B         | 2.61B         | 3.52B          | 2025/01/11   |
| Lumina-Image 2.0 <d-cite key="qinLuminaImage20Unified2025"></d-cite>         | TBD           | TBD           | TBD            | 2025/01/22   |
| SANA 1.5 <d-cite key="xieSANA15Efficient2025a"></d-cite>                     | TBD           | TBD           | TBD            | 2025/03/21   |
| HiDream-I1-Dev <d-cite key="caiHiDreamI1HighEfficientImage2025"></d-cite>    | TBD           | TBD           | TBD            | 2025/04/06   |
| CogView4-6B <d-cite key="zhengCogView3FinerFaster2024a"></d-cite>            | 3.50B         | 2.00B         | 6.00B          | 2025/05/03   |
| Qwen-Image <d-cite key="wu2025qwenimagetechnicalreport"></d-cite>            | 20.43B        | 8.29B         | 28.85B         | 2025/08/04   |

## Pre-trained Text-to-Image Checkpoints

The landscape of pre-trained text-to-image models has evolved dramatically since the introduction of Stable Diffusion. These models serve as powerful foundation models that can be adapted for specialized downstream tasks without architectural modifications, simply by fine-tuning on domain-specific datasets.

## Interactive Architecture Explorer

<div class="l-body">
  <iframe id="architecture-explorer-iframe" src="{{ 'assets/html/2026-04-27-diffusion-architecture-evolution/model-architecture-explorer.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%" style="border: 1px solid #ddd; border-radius: 4px; min-height: 600px;"></iframe>
</div>

<script>
  // Listen for resize messages from the iframe
  window.addEventListener('message', function(e) {
    if (e.data && e.data.type === 'resize') {
      var iframe = document.getElementById('architecture-explorer-iframe');
      if (iframe) {
        iframe.style.height = e.data.height + 'px';
      }
    }
  });
</script>

### U-Net Family

**Stable Diffusion** <d-cite key="rombachHighResolutionImageSynthesis2022"></d-cite> represents the pioneering work in latent diffusion models, adopting a U-Net architecture that operates in a compressed latent space rather than pixel space. This design choice dramatically reduces computational costs while maintaining high-quality generation capabilities. The model combines two key components: a pre-trained variational autoencoder (VAE) for efficient image compression and decompression, and a diffusion model that performs the denoising process in this latent space.

{% include figure.liquid path="assets/img/2026-04-27-diffusion-architecture-evolution/sd.png" alt="Stable Diffusion 1.x - 2.x architecture." caption='Stable Diffusion 1.x - 2.x architecture. Image Credit: <d-cite key="esserTamingTransformersHighResolution2021"></d-cite>.' %}

**Stable Diffusion XL (SDXL)** <d-cite key="podellSDXLImprovingLatent2024"></d-cite> marked a significant scaling advancement, adopting a two-stage U-Net architecture and increasing the model size from 0.8 billion to 2.6 billion parameters. SDXL remains one of the largest U-Net-based models for image generation and demonstrates improved efficiency and compatibility across diverse domains and tasks. Despite reaching scaling limits, SDXL continues to serve as a foundation for numerous specialized applications.

{% include figure.liquid path="assets/img/2026-04-27-diffusion-architecture-evolution/sdxl.png" alt="SDXL Architecture" caption='SDXL Architecture. Image Credit: <d-cite key="podellSDXLImprovingLatent2024"></d-cite>.' %}

**Kandinsky** <d-cite key="razzhigaevKandinskyImprovedTexttoImage2023"></d-cite> represents a significant advancement in the U-Net era, introducing a novel exploration of latent diffusion architecture that combines image prior models with latent diffusion techniques. The model features a modified MoVQ implementation as the image autoencoder component and achieves a FID score of 8.03 on the COCO-30K dataset, marking it as the top open-source performer in terms of measurable image generation quality. **Kandinsky 3** <d-cite key="arkhipkinKandinsky30Technical2024"></d-cite><d-cite key="arkhipkinKandinsky3TexttoImage2024"></d-cite> continues this series with improved text understanding and domain-specific performance, presenting a multifunctional generative framework supporting text-guided inpainting/outpainting, image fusion, and image-to-video generation.

**Stable Cascade** (based on Würstchen architecture) <d-cite key="perniasWurstchenEfficientArchitecture2024"></d-cite> introduces an efficient architecture for large-scale text-to-image diffusion models, achieving competitive performance with unprecedented cost-effectiveness. The key innovation is a latent diffusion technique that learns extremely compact semantic image representations, reducing computational requirements significantly—training requires only 24,602 A100-GPU hours compared to Stable Diffusion 2.1's 200,000 GPU hours while maintaining state-of-the-art results.

**UniDiffuser** <d-cite key="baoOneTransformerFits2023"></d-cite> explores transformer-based diffusion models with a unified framework that fits all distributions relevant to multi-modal data in one model. While primarily focused on transformer architectures, this work demonstrates the potential for unified multi-modal generation within the diffusion framework.

### Pixart-$\alpha$ (2023/10/06)

PixArt-$\alpha$ is motivated by the rising compute and environmental costs of text-to-image systems, seeking near-commercial quality with a much smaller training budget <d-cite key="chenPixArtaFastTraining2024"></d-cite>. In contrast to SD 1.5/2.1, it adopts a large-language-model text encoder (T5) <d-cite key="raffelExploringLimitsTransfer2020"></d-cite>, making it the first open-source diffusion T2I model to use an LLM-based text encoder while keeping the overall design streamlined.

Architecturally, PixArt-$\alpha$ is a latent Diffusion Transformer (DiT): VAE latents are patchified into a token sequence processed by stacked Transformer blocks; each block applies cross-attention to text tokens, and timestep conditioning is injected via a shared adaLN-single, simplifying parameters and conditioning pathways <d-cite key="chenPixArtaFastTraining2024"></d-cite>.

<div style="border: 2px solid #ff9800; background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 15px 0;">
<strong>Key differences vs SD 1.5/2.1</strong>
<ul style="margin-top: 10px; margin-bottom: 0;">
<li>Transformer sequence-of-patches backbone (no encoder–decoder or skip connections)</li>
<li>Shared adaLN for time and unified per-block cross-attention (vs U-Net residual blocks with per-block time MLP/spatial injections)</li>
<li>T5 text encoder (LLM) rather than CLIP/OpenCLIP</li>
</ul>
</div>

{% include figure.liquid path="assets/img/2026-04-27-diffusion-architecture-evolution/pixart_cost.png" alt="Cost Comparison" caption='Comparisons of CO2 emissions and training cost among T2I generators. PIXART-α achieves an exceptionally low training cost of $28,400. Compared to RAPHAEL <d-cite key="xueRAPHAELTexttoImageGeneration2023a"></d-cite>, our CO2 emissions and training costs are merely 1.2% and 0.91%, respectively. Image Credit: <d-cite key="chenPixArtaFastTraining2024"></d-cite>.' %}

{% include figure.liquid path="assets/img/2026-04-27-diffusion-architecture-evolution/pixart_alpha.png" alt="Pixart-α Architecture" caption='Model architecture of PIXART-α. A cross-attention module is integrated into each block to inject textual conditions. To optimize efficiency, all blocks share the same adaLN-single parameters for time conditions. Image Credit: <d-cite key="chenPixArtaFastTraining2024"></d-cite>.' %}

### Lumina-T2I (2024/04/01)

Lumina-T2I is the first entry in the Lumina series from Shanghai AI Lab, aiming for a simple, scalable framework that supports flexible resolutions while maintaining photorealism. Building on the Sora insight that scaling Diffusion Transformers enables generation across arbitrary aspect ratios and durations yet lacks concrete implementation details, Lumina-T2I adopts flow matching to stabilize and accelerate training <d-cite key="gaoLuminaT2XScalableFlowbased2025a"></d-cite>.

Architecturally, Lumina-T2I uses a Flow-based Large Diffusion Transformer (Flag-DiT) with zero-initialized attention, RoPE <d-cite key="suRoFormerEnhancedTransformer2024"></d-cite>, and KQ-Norm <d-cite key="henryQueryKeyNormalizationTransformers2020"></d-cite>. Latent features are tokenized and processed by Transformer blocks; learnable placeholders such as the [nextline] token and layerwise relative position injection enable robust resolution extrapolation without retraining for each size.

<div style="border: 2px solid #ff9800; background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 15px 0;">
<strong>Key differences vs PixArt-α</strong>
<ul style="margin-top: 10px; margin-bottom: 0;">
<li>Robust resolution generalization across 512²–1792²</li>
<li>Uses one-dimensional RoPE, [nextline] token, and layerwise relative position injection</li>
<li>PixArt-α uses absolute positional embeddings limited to the initial layer, degrading at out-of-distribution scales</li>
</ul>
</div>

{% include figure.liquid path="assets/img/2026-04-27-diffusion-architecture-evolution/lumina_t2x.png" alt="Lumina-T2I Architecture" caption='Lumina-T2I architecture featuring Flag-DiT backbone. Image Credit: <d-cite key="gaoLuminaT2XScalableFlowbased2025a"></d-cite>.' %}

### PixArt-$\Sigma$ (2024/04/11)

{% include figure.liquid path="assets/img/2026-04-27-diffusion-architecture-evolution/pixart_sigma.png" alt="PixArt-Σ Architecture" caption='PixArt-Σ architecture and training methodology. Image Credit: <d-cite key="chenPIXARTSWeaktoStrongTraining2024a"></d-cite>.' %}

PixArt-$\Sigma$ achieves superior image quality and user prompt adherence capabilities with significantly smaller model size (0.6B parameters) than existing text-to-image diffusion models, such as SDXL (2.6B parameters) and SD Cascade (5.1B parameters). Moreover, PixArt-$\Sigma$'s capability to generate 4K images supports the creation of high-resolution posters and wallpapers, efficiently bolstering the production of high-quality visual content in industries such as film and gaming.

Architecturally, PixArt-$\Sigma$ maintains the same DiT backbone as PixArt-$\alpha$ but introduces efficient token compression through a novel attention module within the DiT framework that compresses both keys and values, significantly improving efficiency and facilitating ultra-high-resolution image generation. The model incorporates superior-quality image data paired with more precise and detailed image captions, along with data curriculum strategies for improved training effectiveness.

<div style="border: 2px solid #ff9800; background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 15px 0;">
<strong>Key differences vs PixArt-α</strong>
<ul style="margin-top: 10px; margin-bottom: 0;">
<li>Efficient token compression via novel attention module compressing keys and values</li>
<li>Superior-quality training data with more precise and detailed captions</li>
<li>Data curriculum strategies for improved training effectiveness</li>
<li>4K image generation capability for high-resolution content creation</li>
</ul>
</div>

### Lumina-Next-T2I (2024/05/12)

{% include figure.liquid path="assets/img/2026-04-27-diffusion-architecture-evolution/lumina_next.png" alt="Lumina-Next-T2I Architecture" caption='Lumina-Next-T2I Next-DiT architecture. Image Credit: <d-cite key="zhuoLuminaNextMakingLuminaT2X2024a"></d-cite>.' %}

Lumina-Next-T2I <d-cite key="zhuoLuminaNextMakingLuminaT2X2024a"></d-cite> targets the core limitations observed in Lumina-T2X—training instability, slow inference, and resolution extrapolation artifacts—by delivering stronger quality and faster sampling while improving zero-shot multilingual understanding. Unlike prior T2I works that rely on CLIP or T5 encoders <d-cite key="raffelExploringLimitsTransfer2020"></d-cite>, the Lumina series adopts decoder-only LLMs as text encoders: Lumina-T2X uses LLaMA-2 7B <d-cite key="touvronLlama2OpenFoundation2023"></d-cite>, whereas Lumina-Next employs the lighter Gemma-2B to reduce memory and increase throughput. In practice, Lumina-Next shows clear gains on multilingual prompts (vs. CLIP/T5 setups) and further improves text-image alignment with alternative LLMs like Qwen-1.8B and InternLM-7B.

Architecturally, Lumina-Next introduces the Next-DiT backbone with 3D RoPE and Frequency- and Time-Aware Scaled RoPE for robust resolution extrapolation <d-cite key="suRoFormerEnhancedTransformer2024"></d-cite>. It adds sandwich normalizations to stabilize training (cf. normalization strategies such as KQ-Norm <d-cite key="henryQueryKeyNormalizationTransformers2020"></d-cite>), a sigmoid time discretization schedule to reduce Flow-ODE sampling steps, and a Context Drop mechanism that merges redundant visual tokens to accelerate inference—all while retaining the flow-based DiT formulation of the Lumina family.

<div style="border: 2px solid #ff9800; background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 15px 0;">
<strong>Key differences vs Lumina-T2I</strong>
<ul style="margin-top: 10px; margin-bottom: 0;">
<li>Next-DiT with 3D RoPE + frequency/time-aware scaling for stronger resolution extrapolation</li>
<li>Sandwich normalizations improve stability; sigmoid time schedule reduces sampling steps</li>
<li>Context Drop merges redundant tokens for faster inference throughput</li>
<li>Decoder-only LLM text encoders (Gemma-2B by default; Qwen-1.8B/InternLM-7B optional) boost zero-shot multilingual alignment vs CLIP/T5</li>
</ul>
</div>

### Stable Diffusion 3 (2024/06/12)

Stable Diffusion 3 aims to improve existing noise sampling techniques for training rectified flow models by biasing them towards perceptually relevant scales, demonstrating superior performance compared to established diffusion formulations for high-resolution text-to-image synthesis <d-cite key="esserScalingRectifiedFlow2024"></d-cite>. This work presents the first comprehensive scaling study for text-to-image DiTs, establishing predictable scaling trends and correlating lower validation loss to improved synthesis quality across various metrics and human evaluations.

Architecturally, SD3 transitions from DiT's cross-attention blocks to MMDiT (Multimodal Diffusion Transformer) with double-stream blocks that use separate weights for the two modalities, enabling bidirectional flow of information between image and text tokens for improved text comprehension and typography. Unlike SDXL which relies primarily on CLIP encoders, SD3 incorporates both CLIP (L/14 and OpenCLIP bigG/14) and T5-XXL encoders <d-cite key="raffelExploringLimitsTransfer2020"></d-cite>, concatenating pooled outputs and hidden representations to create comprehensive text conditioning with enhanced understanding capabilities.

<div style="border: 2px solid #ff9800; background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 15px 0;">
<strong>Key differences vs SDXL and PixArt-α</strong>
<ul style="margin-top: 10px; margin-bottom: 0;">
<li>MMDiT double-stream architecture with separate weights per modality and bidirectional information flow (vs single-stream cross-attention)</li>
<li>Integrated rectified flow training with perceptually-biased noise sampling (vs standard diffusion formulation)</li>
<li>Combined CLIP + T5-XXL text encoding for enhanced text comprehension and typography</li>
<li>First comprehensive scaling study demonstrating predictable trends for text-to-image DiTs</li>
</ul>
</div>

{% include figure.liquid path="assets/img/2026-04-27-diffusion-architecture-evolution/sd3.png" alt="Stable Diffusion 3 Architecture" caption='Stable Diffusion 3 MMDiT architecture. Image Credit: <d-cite key="esserScalingRectifiedFlow2024"></d-cite>.' %}

### Flux.1-Dev (2024/08/02)

Flux.1-Dev, developed by former Stability AI core members, aims to scale beyond previous models and achieve superior image quality with more accurate text-to-image synthesis <d-cite key="blackforestlabsFLUX1"></d-cite>. Representing a significant scaling effort, the model features a massive 12 billion parameter generator combined with a 4.7 billion parameter text encoder, marking substantial growth compared to predecessors and establishing new benchmarks in AI-driven image generation capabilities.

Architecturally, Flux.1-Dev advances beyond SD3's MMDiT by implementing a hybrid architecture that combines both single-stream and double-stream Multi-Modal Diffusion Transformers, enhancing the model's ability to process complex visual-textual relationships. Like SD3, it incorporates T5 text encoding <d-cite key="raffelExploringLimitsTransfer2020"></d-cite> and integrates rectified flow techniques for more stable and efficient training, while conducting a comprehensive scaling study that optimizes performance across the substantially larger parameter space.

<div style="border: 2px solid #ff9800; background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 15px 0;">
<strong>Key differences vs SD3</strong>
<ul style="margin-top: 10px; margin-bottom: 0;">
<li>Hybrid single-stream + double-stream MMDiT architecture (vs purely double-stream MMDiT)</li>
<li>Massive scaling to 12B generator + 4.7B text encoder parameters (vs smaller SD3 variants)</li>
<li>Enhanced rectified flow implementation optimized for larger scale training</li>
<li>Comprehensive scaling study specifically designed for multi-billion parameter DiTs</li>
</ul>
</div>

{% include figure.liquid path="assets/img/2026-04-27-diffusion-architecture-evolution/flux_dit.png" alt="Flux.1-Dev Architecture" caption='Flux.1-Dev MMDiT architecture. Image Credit: <d-cite key="blackforestlabsFLUX1"></d-cite>.' %}

### CogView3 & CogView3-Plus (2024/10/13)

{% include figure.liquid path="assets/img/2026-04-27-diffusion-architecture-evolution/cogview3.png" alt="CogView3 Architecture." caption='(left) The pipeline of CogView3. User prompts are rewritten by a text-expansion language model. The base stage model generates 512 × 512 images, and the second stage subsequently performs relaying super-resolution. (right) Formulation of relaying super-resolution in the latent space. Image Credit: <d-cite key="zhengCogView3FinerFaster2024a"></d-cite>.' %}

**CogView3** <d-cite key="zhengCogView3FinerFaster2024a"></d-cite> introduces a **relay diffusion approach** <d-cite key="tengRelayDiffusionUnifying2024"></d-cite> that generates low-resolution images first, then refines them through super-resolution to achieve 2048×2048 outputs. This multi-stage process reduces computational costs while improving quality—CogView3 outperformed SDXL by 77% in human evaluations while using only one-tenth the inference time. The model employs a text-expansion language model to rewrite user prompts, with a base stage generating 512×512 images followed by relaying super-resolution in the latent space.

**CogView3-Plus** upgrades to DiT architecture with Zero-SNR scheduling and joint text-image attention for further efficiency gains. This architectural evolution represents a significant step in the CogView series, transitioning from traditional approaches to transformer-based diffusion models while maintaining the efficiency advantages of the relay diffusion framework.

### Hunyuan-DiT (2024/12/01)

Hunyuan-DiT, developed by Tencent's Hunyuan team, aims to create a powerful multi-resolution diffusion transformer capable of fine-grained understanding of both English and Chinese languages, addressing the need for state-of-the-art Chinese-to-image generation with culturally relevant and multilingual capabilities <d-cite key="liHunyuanDiTPowerfulMultiResolution2024a"></d-cite>. The model establishes a comprehensive data pipeline with iterative optimization, employing a Multimodal Large Language Model to refine image captions and enhance alignment between textual descriptions and generated images, particularly for intricate Chinese characters and cultural nuances.

Architecturally, Hunyuan-DiT builds upon PixArt-$\alpha$ by incorporating both single-stream and double-stream Multi-Modal Diffusion Transformer (MM-DiT) blocks similar to SD3, enabling efficient handling of complex image generation tasks across multiple resolutions. The model integrates dual text encoders—CLIP for understanding overall semantic content and T5 <d-cite key="raffelExploringLimitsTransfer2020"></d-cite> for nuanced language comprehension including complex sentence structures—combined with enhanced positional encoding to maintain spatial information across different resolutions, facilitating robust multi-resolution generation capabilities.

<div style="border: 2px solid #ff9800; background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 15px 0;">
<strong>Key differences vs PixArt-α</strong>
<ul style="margin-top: 10px; margin-bottom: 0;">
<li>Single-stream + double-stream MM-DiT blocks for enhanced multi-modal processing (vs single-stream cross-attention)</li>
<li>Dual text encoders (CLIP + T5) for semantic and nuanced language understanding (vs T5 only)</li>
<li>Multi-resolution diffusion transformer with enhanced positional encoding for robust resolution handling</li>
<li>Multimodal LLM-refined captions with fine-grained bilingual (English + Chinese) understanding</li>
</ul>
</div>

{% include figure.liquid path="assets/img/2026-04-27-diffusion-architecture-evolution/hunyuandit.png" alt="Hunyuan-DiT Architecture" caption='Hunyuan-DiT multi-resolution architecture. Image Credit: <d-cite key="liHunyuanDiTPowerfulMultiResolution2024a"></d-cite>.' %}

### SANA (2025/01/11)

SANA, developed by NVIDIA, aims to enable efficient high-resolution image synthesis up to 4096×4096 pixels while maintaining deployment feasibility on consumer hardware, generating 1024×1024 images in under a second on a 16GB laptop GPU <d-cite key="xieSANAEfficientHighResolution2025"></d-cite>. The model introduces innovations to reduce computational requirements dramatically: DC-AE (deep compression autoencoder) achieves 32× image compression reducing latent tokens significantly, efficient caption labeling and selection accelerate convergence, and Flow-DPM-Solver reduces sampling steps for faster generation.

Architecturally, SANA advances beyond PixArt-$\Sigma$ by replacing traditional self-attention mechanisms with Linear Diffusion Transformer (Linear DiT) blocks, enhancing computational efficiency at high resolutions without compromising quality. The model adopts a decoder-only small language model as the text encoder, employing complex human instructions with in-context learning to improve text-image alignment compared to conventional CLIP or T5 encoders. The compact 0.6B parameter model achieves competitive performance with substantially larger models like Flux-12B while being 20 times smaller and over 100 times faster in throughput.

<div style="border: 2px solid #ff9800; background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 15px 0;">
<strong>Key differences vs PixArt-Σ</strong>
<ul style="margin-top: 10px; margin-bottom: 0;">
<li>Linear DiT replacing traditional self-attention for O(n) complexity vs O(n²) at high resolutions</li>
<li>DC-AE with 32× compression reducing latent tokens and memory requirements dramatically</li>
<li>Decoder-only language model as text encoder with in-context learning (vs T5)</li>
<li>0.6B parameters achieving competitive quality with 12B models while 100× faster throughput</li>
</ul>
</div>

{% include figure.liquid path="assets/img/2026-04-27-diffusion-architecture-evolution/sana.png" alt="SANA Architecture" caption='SANA Linear DiT architecture for efficient high-resolution generation. Image Credit: <d-cite key="xieSANAEfficientHighResolution2025"></d-cite>.' %}

### Lumina-Image 2.0 (2025/01/22)

Lumina-Image 2.0 aims to provide a unified and efficient image generative framework that excels in generating high-quality images with strong text-image alignment across diverse generation and editing tasks <d-cite key="qinLuminaImage20Unified2025"></d-cite>. Building upon the Lumina series' foundation, the model consolidates multiple generation tasks into a cohesive framework, optimizing performance and efficiency to cater to a wide range of image generation applications while achieving competitive scores across multiple benchmarks including FID and CLIP metrics.

Architecturally, Lumina-Image 2.0 advances beyond Lumina-Next-T2I by introducing a unified Next-DiT architecture that seamlessly integrates text-to-image generation and image editing capabilities within a shared framework. The model maintains the Lumina series' architectural strengths including 3D RoPE <d-cite key="suRoFormerEnhancedTransformer2024"></d-cite>, frequency-aware scaling, and flow-based formulation, while enhancing the framework to support both generation and editing operations efficiently. This unified approach enables the model to leverage shared representations and training strategies across different image generation modalities.

<div style="border: 2px solid #ff9800; background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 15px 0;">
<strong>Key differences vs Lumina-Next-T2I</strong>
<ul style="margin-top: 10px; margin-bottom: 0;">
<li>Unified Next-DiT framework seamlessly integrating generation and editing (vs generation-only focus)</li>
<li>Enhanced multi-task architecture supporting diverse image generation applications within single model</li>
<li>Optimized training paradigm leveraging shared representations across generation modalities</li>
<li>Competitive performance across FID and CLIP benchmarks with improved efficiency</li>
</ul>
</div>

{% include figure.liquid path="assets/img/2026-04-27-diffusion-architecture-evolution/lumina_image2.png" alt="Lumina-Image 2.0 Architecture" caption='Lumina-Image 2.0 Unified Next-DiT architecture. Image Credit: <d-cite key="qinLuminaImage20Unified2025"></d-cite>.' %}

### SANA 1.5 (2025/03/21)

SANA 1.5 aims to push the boundaries of efficient high-resolution image synthesis established by SANA, offering improved performance and scalability through larger model sizes and advanced inference scaling techniques <d-cite key="xieSANA15Efficient2025a"></d-cite>. The model introduces inference scaling via VISA (a specialized NVILA-2B model) that scores and selects top images from large candidate sets, significantly boosting GenEval performance scores—for instance, improving SANA-1.5-4.8B from 81 to 96. This approach demonstrates that post-generation selection can dramatically enhance quality metrics without architectural changes.

Architecturally, SANA 1.5 builds upon the original SANA by incorporating an enhanced DC-AE (deep compression autoencoder) to handle higher resolutions and more complex generation tasks, along with advanced Linear DiT blocks featuring more sophisticated linear attention mechanisms to boost efficiency and quality in high-resolution synthesis. The model scales to 4.8B parameters compared to SANA's 0.6B, providing a robust solution for generating high-quality images with strong text-image alignment suitable for diverse professional applications requiring both quality and computational efficiency.

<div style="border: 2px solid #ff9800; background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 15px 0;">
<strong>Key differences vs SANA</strong>
<ul style="margin-top: 10px; margin-bottom: 0;">
<li>Inference scaling with VISA model for candidate selection dramatically improving GenEval scores (81→96)</li>
<li>Enhanced DC-AE handling higher resolutions and more complex generation tasks</li>
<li>Advanced Linear DiT with more sophisticated linear attention mechanisms</li>
<li>Scaled to 4.8B parameters providing improved quality while maintaining efficiency advantages</li>
</ul>
</div>

{% include figure.liquid path="assets/img/2026-04-27-diffusion-architecture-evolution/sana1_5.png" alt="SANA 1.5 Architecture" caption='SANA 1.5 improved Linear DiT architecture. Image Credit: <d-cite key="xieSANA15Efficient2025a"></d-cite>.' %}

### HiDream-I1-Dev (2025/04/06)

HiDream-I1, developed by HiDream.ai, addresses the critical trade-off between quality improvements and computational complexity in image generative foundation models, aiming to achieve state-of-the-art image generation quality within seconds while maintaining high efficiency <d-cite key="caiHiDreamI1HighEfficientImage2025"></d-cite>. With 17 billion parameters, the model introduces a sparse Diffusion Transformer structure that enables efficient inference suitable for professional-grade design needs, supporting 4K ultra-high-definition image generation with advanced text comprehension, multi-style adaptation, and precise detail control while optimizing computational requirements through sparsity.

Architecturally, HiDream-I1 advances beyond Flux.1-Dev and Qwen-Image by implementing a novel sparse DiT structure where only subsets of transformer blocks are activated for each forward pass, dramatically reducing computational costs while maintaining generation quality. The sparse architecture enables the massive 17B parameter model to achieve practical inference speeds comparable to smaller dense models, with efficient diffusion mechanisms supporting multimodal input and providing fine-grained control over generation. This sparse approach represents a paradigm shift in scaling DiT models, demonstrating that architectural efficiency through sparsity can rival quality of substantially denser models.

<div style="border: 2px solid #ff9800; background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 15px 0;">
<strong>Key differences vs Flux.1-Dev and other large DiTs</strong>
<ul style="margin-top: 10px; margin-bottom: 0;">
<li>Sparse DiT structure activating only subsets of blocks per forward pass for efficient 17B parameter model</li>
<li>4K ultra-high-definition generation support with optimized inference speed despite massive scale</li>
<li>Advanced sparse attention mechanisms maintaining quality while dramatically reducing computational costs</li>
<li>Multimodal input support and fine-grained control optimized for professional-grade design applications</li>
</ul>
</div>

{% include figure.liquid path="assets/img/2026-04-27-diffusion-architecture-evolution/hidream.png" alt="HiDream-I1-Dev Architecture" caption='HiDream-I1-Dev Sparse DiT architecture. Image Credit: <d-cite key="caiHiDreamI1HighEfficientImage2025"></d-cite>.' %}

### CogView4-6B (2025/05/03)

**CogView4-6B** <d-cite key="zhengCogView3FinerFaster2024a"></d-cite> represents the latest advancement in the CogView series, featuring a sophisticated **CogView4Transformer2DModel** architecture that excels in Chinese text rendering and multilingual image generation. The model demonstrates exceptional performance in text accuracy evaluation, achieving precision of 0.6969, recall of 0.5532, and F1 score of 0.6168 on Chinese text benchmarks.

CogView4-6B leverages GLM-based text encoding and advanced transformer blocks with RoPE (Rotary Position Embedding) for enhanced spatial understanding and text-image alignment. This architectural sophistication enables the model to achieve superior text rendering capabilities, particularly for complex Chinese characters and multilingual content, setting new standards for text-to-image generation in non-Latin scripts. Available on [Hugging Face](https://huggingface.co/zai-org/CogView4-6B) under Apache 2.0 license.

### Qwen-Image (2025/08/04)

Qwen-Image addresses critical challenges in text-to-image generation where even state-of-the-art commercial models struggle with complex prompt alignment—particularly multi-line text rendering, non-alphabetic languages (e.g., Chinese), localized text insertions, and seamless text-visual integration <d-cite key="wu2025qwenimagetechnicalreport"></d-cite>. Through comprehensive data engineering with multi-stage filtering pipelines, progressive curriculum learning from basic text rendering to paragraph-level layouts, and post-training refinements including supervised fine-tuning (SFT), direct preference optimization (DPO) reinforcement learning, and general reward preference optimization (GRPO), Qwen-Image achieves superior text rendering capabilities across both alphabetic and logographic languages with high fidelity.

Architecturally, Qwen-Image advances beyond Flux.1-Dev by adopting a Vision-Language Model (VLM) as its text encoder—specifically Qwen2.5-VL 7B—enabling richer semantic understanding compared to traditional text-only encoders. The model employs Multimodal Scalable RoPE for robust position encoding and integrates dual-conditioning through MMDiT architecture: semantic features extracted via Qwen-VL capture high-level scene understanding, while reconstructive features from the VAE encoder preserve low-level visual details, jointly fed as conditioning signals for enhanced text-image alignment and visual consistency.

<div style="border: 2px solid #ff9800; background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 15px 0;">
<strong>Key differences vs Flux.1-Dev</strong>
<ul style="margin-top: 10px; margin-bottom: 0;">
<li>VLM-based text encoder (Qwen2.5-VL 7B) capturing visual-semantic understanding (vs text-only T5 encoder)</li>
<li>Multimodal Scalable RoPE for enhanced position encoding across text and visual modalities</li>
<li>Dual-conditioning design with semantic features (Qwen-VL) and reconstructive features (VAE) for superior text rendering</li>
<li>Multi-stage training with curriculum learning, SFT, DPO RL, and GRPO RL for complex text alignment</li>
</ul>
</div>

{% include figure.liquid path="assets/img/2026-04-27-diffusion-architecture-evolution/qwen_image.png" alt="Qwen-Image Architecture" caption='Qwen-Image DiT architecture with advanced text rendering capabilities. Image Credit: <d-cite key="wu2025qwenimagetechnicalreport"></d-cite>.' %}

## Experiments and Case Studies

To comprehensively evaluate the capabilities of different text-to-image diffusion models, we propose a systematic evaluation framework spanning tasks of varying complexity. This section will present case studies of text-to-image generation visualizations using existing checkpoints, assessing their performance across a spectrum of increasingly challenging tasks.

### Task Hierarchy: From Simple to Complex

#### 1. **Simple Object and Scene Generation**

- **Objective**: Generate images of single objects or simple scenes based on straightforward prompts
- **Complexity**: Low
- **Evaluation Criteria**: Object-level generalization and prompt following
- **Requirements**: No restrictions on image style; focus on basic visual understanding
- **Example Prompts**: "A red apple on a wooden table", "A cat sitting in sunlight"

#### 2. **Multi-Object Generation with Relationship Understanding**

- **Objective**: Generate images featuring multiple objects with specified spatial and semantic relationships
- **Complexity**: Medium
- **Evaluation Criteria**: Accuracy in depicting object relationships and interactions
- **Requirements**: Understanding of spatial prepositions, relative positioning, and object interactions
- **Example Prompts**: "A dog chasing a ball in a park", "A book next to a coffee cup on a desk"

#### 3. **World Knowledge Generation**

- **Objective**: Generate images that require incorporation of diverse world knowledge across multiple domains
- **Complexity**: High
- **Evaluation Criteria**: Accurate representation of real-world knowledge, cultural understanding, and contextual appropriateness
- **Requirements**: Understanding of geography, history, culture, science, and general world facts
- **Example Prompts**: "The Eiffel Tower during cherry blossom season", "A traditional Japanese tea ceremony in a modern setting"

#### 4. **Text Rendering**

- **Objective**: Generate images that include accurate, legible text within the visual composition
- **Complexity**: Very High
- **Evaluation Criteria**: Text clarity, spelling accuracy, appropriate integration with visual elements
- **Requirements**: Understanding of typography, language structure, and visual-text integration
- **Example Prompts**: "A storefront with a 'COFFEE SHOP' sign", "A book cover with the title 'Machine Learning Fundamentals'"

#### 5. **Reasoning and Compositional Understanding**

- **Objective**: Generate images based on prompts requiring logical reasoning, inference, and complex compositional understanding
- **Complexity**: Extremely High
- **Evaluation Criteria**: Logical consistency, compositional coherence, and reasoning accuracy
- **Requirements**: Multi-step reasoning, causal understanding, and complex scene composition
- **Example Prompts**: "The aftermath of a tea party in a garden", "What would happen if gravity worked sideways in an office"

### Evaluation Methodology

*[Placeholder for detailed evaluation results and model comparisons across these task categories]*

## Discussion

### Diffusion Models vs. Auto-regressive Models

Diffusion models and autoregressive (AR) models represent two fundamentally different approaches to image generation, with the key distinction being that **autoregressive models operate on discrete image tokens** while **diffusion models work with continuous representations**. Autoregressive models like DALL-E <d-cite key="rameshZeroShotTexttoImageGeneration2021"></d-cite>, CogView <d-cite key="dingCogViewMasteringTexttoImage2021a"></d-cite>, and CogView2 <d-cite key="dingCogView2FasterBetter2022"></d-cite> treat image generation as a sequence modeling problem, encoding images into discrete tokens using VQ-VAE <d-cite key="esserTamingTransformersHighResolution2021"></d-cite> or similar vector quantization methods, then autoregressively predicting the next token given previous tokens. This approach offers sequential generation with precise control and natural language integration, but suffers from slow generation, error accumulation, and discrete representation loss. In contrast, diffusion models operate directly on continuous pixel or latent representations, learning to reverse a gradual noise corruption process, which enables parallel generation, high-quality outputs, and flexible conditioning, though at the cost of computational overhead and less direct control. Recent advances have significantly improved autoregressive approaches: VAR <d-cite key="tianVisualAutoregressiveModeling2024"></d-cite> redefines autoregressive learning as coarse-to-fine "next-scale prediction" and achieves superior performance compared to diffusion transformers, while Infinity <d-cite key="hanInfinityScalingBitwise2025"></d-cite> demonstrates effective scaling of bitwise autoregressive modeling for high-resolution synthesis. Additionally, MAR <d-cite key="liAutoregressiveImageGeneration2024a"></d-cite> bridges the gap between paradigms by adopting diffusion loss for autoregressive models, enabling continuous-valued autoregressive generation without vector quantization. Recent work has also explored hybrid approaches that combine both paradigms: HunyuanImage 3.0 <d-cite key="caoHunyuanImage30Technical2025"></d-cite> and BLIP3-o <d-cite key="chenBLIP3oFamilyFully2025"></d-cite> demonstrate unified multimodal models within autoregressive frameworks while incorporating diffusion-inspired techniques, while OmniGen <d-cite key="xiaoOmniGenUnifiedImage2024"></d-cite> and OmniGen2 <d-cite key="wuOmniGen2ExplorationAdvanced2025"></d-cite> use diffusion models as backbones for unified generation capabilities.