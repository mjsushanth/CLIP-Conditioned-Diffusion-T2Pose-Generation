
# Technical Deep-Dive: Research Insights & Non-Obvious Discoveries

## The UNet-Transformer Bridge

### The Fundamental Tension
The most challenging aspect wasn't implementing individual components but creating a bridge between two fundamentally different architectural paradigms. CNNs expect channel-first tensors `[B, C, H, W]` while transformers work with sequence-last features `[B, seq, features]`. Realized that LayerNorm could serve as the universal normalization strategy, operating on the last dimension regardless of tensor shape.
Initially attempted GroupNorm with complex reshape operations to maintain CNN compatibility. The failure wasn't just about tensor shapes— GroupNorm fundamentally assumes channel-grouped feature statistics, which breaks when cross-attention outputs don't follow CNN channel semantics. 
LayerNorm's last-dimension operation naturally accommodates both `[B, features]` pose vectors and `[B, seq, features]` attention outputs without dimensional gymnastics.

### Cross-Attention Integration
Cross-attention placement within residual blocks. The pattern emerged: `norm → conv → time_embedding → attention → norm → conv → residual`. 
This sequence allows partially-processed features to query text embeddings before final transformation, creating a semantic modulation point that preserves both spatial structure and linguistic guidance.

## Pre-Normalization: 
### Why Pre-Norm Outperformed Post-Norm
Traditional post-normalization `conv → norm → activation` created unstable gradients in diffusion training. Pre-normalization `norm → activation → conv` ensures clean feature distributions enter each operation. 
In diffusion models, where gradients flow through effectively thousands of unrolled timesteps, this stability difference becomes critical for convergence.

### Time Embedding Injection Strategy
Time embeddings placement in the middle of residual blocks wasn't arbitrary—it creates a feature modulation point. Network can adjust its processing based on noise level. Early injection overwhelms spatial features; late injection lacks influence. Mid-block placement allows time-aware feature transformation without destroying pose structure.


## The Anatomy Constraint 
We had a catastrophic Phase 1 failure stemmed from calculating bone length constraints in normalized space while the dataset was preprocessed differently. The anatomy loss operated on normalized joint positions, but real anatomical relationships exist in metric space. 
The fix required denormalizing poses before constraint calculation—a seemingly obvious detail that caused weeks of debugging. Newbie mistake! :) 

### Forward Kinematics
Initial attempts used direct joint-pair constraints, which created conflicting forces across the skeleton. 
Forward kinematics chains (pelvis → spine → extremities) provided a hierarchical solution where parent corrections propagate to children, maintaining overall skeleton coherence.


## Dual-Pass Classifier-Free Guidance
CFG required running both conditional and unconditional forward passes. The critical insight: null embeddings aren't just zeros—they represent the learned unconditional distribution. 
The guidance formula `uncond + scale * (cond - uncond)` creates a vector in the direction of conditioning, with scale controlling strength.
Discovered that attention weights from unconditional passes contaminated visualization. 
The solution: temporarily disable attention storage during null embedding passes, only capturing weights during conditional generation. 


## The CLIP Projection Layer:
### 512D→256D:
CLIP embeddings contain visual, linguistic, and conceptual information; the projection layer extracts pose-relevant semantics while discarding irrelevant features.
### Global vs Token-Level Conditioning Trade-off
Using CLIP's global text embedding rather than token-level features simplified architecture but limited fine-grained control. 


## Dataset Processing: The First-Action Extraction Algorithm
The dataset contained motion sequences with compound actions. The breakthrough came from using part-of-speech tags to identify conjunction markers (`CCONJ`, `SCONJ`) and temporal adverbs ("then", "after"). 
Truncating at the first conjunction isolated primary actions, creating cleaner text-pose pairs.
Uniform random sampling created bias toward common poses (49.6% cluster dominance). 
Strategic cluster sampling with predetermined quotas ensured rare pose representation during training. 


## Res Block Evolution: From Simple to Conditioned

The solution for changing dimensions: `self.skip_connection = nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels)`. 
This pattern handles both identity and projection cases without manual dimension tracking, maintaining information flow regardless of channel changes.
Conditioned residual blocks integrate three information streams: spatial features, temporal information (timestep), and semantic guidance (text). 
The sequential integration pattern proved critical: features → time → attention → features. This ordering allows each modality to influence processing without overwhelming others.


## Debugging Insights: 
Early generated poses showed extreme spatial distortions. Root cause: misunderstanding HumanML3D's coordinate system where bodies face +Y with left side on +X. 
The visualization fix required coordinate transformation matrices, but more importantly, revealed that the model was learning correct relationships in an unexpected coordinate frame.
Poses appeared to float or have extreme root positions. Solution: explicit pelvis centering during both training and inference. This wasn't just preprocessing—it established a consistent reference frame that allowed the model to focus on pose relationships rather than absolute positioning.
