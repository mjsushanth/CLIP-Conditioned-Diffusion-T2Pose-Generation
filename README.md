# Text-to-Pose Generation with Diffusion Models

## Motivation & Overview

Human movement is one of the most expressive forms of communication, carrying emotional nuance and contextual meaning that transcends spoken language. The challenge of translating natural language descriptions into anatomically plausible human poses represents a fascinating intersection of semantic understanding and structural generation—where AI must bridge the gap between how we describe movement and how bodies actually move through space. This project tackles the fundamental question: can we teach machines to understand pose descriptions well enough to generate realistic human configurations that match our linguistic intentions?

We chose HumanML3D as our foundation because it provides the critical missing link between natural language and motion data—a dataset where every pose sequence is paired with human-written descriptions that capture the nuanced ways people naturally describe movement. Unlike synthetic or procedurally generated datasets, HumanML3D contains the authentic linguistic patterns and anatomical diversity needed to train models that can handle real-world pose generation requests. The dataset's 22-joint representation offers sufficient anatomical detail while remaining computationally tractable, making it ideal for exploring how diffusion models can learn to generate poses that are both semantically aligned with text and anatomically coherent.



# Technical Deep-Dive: Text-to-Pose Generation Pipeline

## 1. Strategic Pose Dataset Engineering & Multi-Cluster Sampling
K-means pose diversity analysis on HumanML3D, 8-cluster balanced sampling strategy, avoided 49.6% cluster dominance bias, POS-tag syntactic segmentation, first-action extraction from sequences, dynamic sampling configs (1.3K-18K), cluster-to-pose mapping architecture, anatomical diversity preservation, rare pose category inclusion, temporal artifact elimination

## 2. Advanced Pose Representation & Coordinate System Normalization
263D HumanML3D vector processing, 66D joint positions extraction, root-pelvis centering normalization, anatomical proportion preservation, SMPL Y-up to Z-up transformation, canonical orientation solving, 22-joint hierarchical mapping, kinematic chain integrity, centroid-distance outlier detection, pelvis-spine-extremity validation

## 3. Hybrid CNN-Transformer Diffusion Architecture
ConditionedUNet design, CNN-transformer bridge, CLIP embedding integration, multi-head cross-attention, LayerNorm paradigm shift, GroupNorm elimination, residual block conditioning, time embedding injection, pre-norm activation pattern, multi-resolution text projection

## 4. CLIP Semantic Encoding & Projection Pipeline
Frozen CLIP-ViT weights, 512D→256D learnable projection, text tokenization preprocessing, embedding normalization, domain adaptation mapping, semantic-to-pose feature bridge, compositional understanding transfer, multi-level conditioning integration, cross-modal representation learning

## 5. Dual-Pass CFG. [ Classifier-Free Guidance. ]
Conditional-unconditional forward passes, null embedding generation, training-time dual inference, sampling-time guidance scaling, progressive guidance ramping (2.0→7.0), attention storage control, text alignment optimization, generation controllability

## 6. Anatomical Constraint Enforcement & Bone Length Consistency
Forward kinematics chains, reference bone length database, pelvis-centered constraint application, scale factor clamping, extremity joint validation, anatomical plausibility scoring, post-generation constraint enforcement, skeleton coherence preservation

## 7. Optimized Noise Scheduling & Diffusion Training
1000-step DDPM scheduler, SquaredCos noise schedule, beta range optimization (1e-4→0.006), mixed precision training, gradient clipping implementation, OneCycleLR scheduling, anatomy loss weighting, multi-epoch checkpoint system

## 8. Evaluation & Visualization Pipeline
Real-time pose rendering, 3D skeleton visualization, text-pose alignment metrics, attention pattern analysis, progressive generation tracking, multi-prompt batch inference, anatomical validation scoring, interactive result exploration



## Results & Performance Analysis

### Quantitative Results Overview
Phase 1 baseline diffusion failed catastrophically, with anatomy loss exceeding 10^16 indicating complete structural breakdown. Phase 2 anatomical improvements achieved stable convergence (final loss: 1.17, anatomy loss: 0.16), - successful skeletal constraint integration. Phase 3 text conditioning delivered optimal performance (final loss: 0.69, anatomy loss: 0.07), with diffusion loss reducing to 0.63 while maintaining anatomical plausibility.

### Semantic Alignment Achievements  
Generated poses demonstrate clear text correspondence, "arms raised" produces elevated limb configurations, "golf swing" exhibits dynamic rotation and weight transfer, "crouching" shows appropriate knee flexion and torso adjustment, cross-attention mechanism successfully bridges CLIP semantic space with pose feature space!
