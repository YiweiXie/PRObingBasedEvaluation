# 🛡️ PROBE: Diagnosing Residual Concept Capacity in Erased Text-to-Video Diffusion Models
<div align="left">

[![Paper](https://img.shields.io/badge/Paper-ArXiv-red.svg)](https://arxiv.org/abs/xxxx.xxxxx) [![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://your-project-page.github.io) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

## 📖 Abstract
This repository contains the official codebase for **PROBE**, a novel diagnostic framework designed to evaluate the true efficacy of concept erasure methods in Text-to-Video (T2V) diffusion models. By introducing a continuous latent optimization strategy via pseudo-words (`<v*>`), PROBE successfully bypasses both input-level conditioning (e.g. NegPrompt), cross-attention steering (e.g., SAFREE) and weight-space unlearning (e.g., T2VUnlearning), revealing severe *Semantic Leakage* in both v-predition and flow matching architectures (e.g., CogVideoX-5B, Wan2.2-5B).

---

## ⚙️ Installation Guide
We strongly recommend running this codebase on a Linux system equipped with NVIDIA A100/H100 GPUs for evaluating large-scale T2V models.

```bash
git clone https://github.com/YiweiXie/PRObingBasedEvaluation.git
cd PRObingBasedEvaluation

conda create -n probe python=3.10
conda activate probe

pip install torch torchvision
pip install -r requirements.txt
```

## 🚀 Quick Start
Our pipeline is strictly designed to ensure maximum reproducibility of the empirical results reported in the main paper.

#### Reference Data Preparation
Before executing the PROBE optimization, a high-quality reference dataset must be curated to provide the spatial-temporal priors for the target concept. This dataset typically consists of some representative video clips that contain the erased concept (e.g., objects, nudity, or identities). 
```bash
 python ./inference/CogX_2b.py
    --csv_path "./evaluation/data/nudity_cogvideox.csv"
    --out_put "./output/cogx2b/nudity"
    --seed 42
    --num_frames 49
    --generate_clean True
    --generate_neg False    # you can set it true as need
    --generate_probe False  # same as above
    --embed_path "" # if you want to generate videos after probe, please provide it
```

#### Automated Video Filtering
To provide high quality reference data. we need to transfer videos to frames and use pre-trained classifiers to automatically filter frames containing the target concept.
```bash
# Transfer videos to frames
 python ./evaluation/mp4_to_png.py
    --input_folder "./ouput/cogx2b/nudity"
    --out_put "./output/cogx2b/nudity_png"
# Selection
 python ./evaluation/q16_nudenet_detect_select.py
    --folder_path "./output/cogx2b/nudity_png"
    --prompt_path "./evaluation/data/nudity_cogvideox.csv"
    --unsafe_output_path "./train_data/cogvideo2bX/nudity"
```

#### Pseudo-word Training
The core entry point for our probe is train.py. This script freezes the targeted T2V model's weights and optimizes the continuous token embeddings to maximize the adversarial objective $\mathcal{L}_{total} = \mathcal{L}_{rec} + \lambda \mathcal{L}_{align}$.
This will train a pseudo-word and save the results accordingly:
```bash
 python train.py 
     --erasure_model "cogvideox2b"
     --concept "nudity"
     --initializer_token "naked"
     --learnable_property "object"
     --neg_prompt "nudity"
     --num 5
     --num_steps 3000
     --train_data_dir "./train_data/cogvideo2bX/nudity"
     --output_dir "./results/probe_nudity" 
```

#### Multi Dimensions Evaluation
To rigorously quantify the reactivation rates and structural integrity of the generated videos, our repository provides a unified evaluation pipeline covering four distinct metrics (Classifier-based, CLIP-based, Temporal reactivation curve and Human validation).
```bash
# Classifier-based Evaluation (For nudity)
  python ./evaluation/q16_nudenet_detect_select.py
    --folder_path "./output/cogx2b/nudity_png"
    --prompt_path "./evaluation/data/nudity_cogvideox.csv"
    --unsafe_output_path "./train_data/cogvideo2bX/nudity"
# CLIP-based Metrics
 python ./evaluation/CLIP-based_Score.py
    --videos_path "./ouput/cogx2b/nudity"
    --output_path "./ouput/cogx2b/clip_eval"
# Temporal reactivation curve
 python ./evaluation/temporal_eval.py
    --base_path "./ouput/cogx2b/nudity_png"
    --type_input "nudity"
    --target_word "nudity"
    --n_frames 49
# Human validation (Prepare blind videos)
 python sample_for_human_validation.py
    --nudity_path "./output/cogx2b/nudity"
    --samples_per_class 3    
```

## 📝 Citation
If you find this codebase or our theoretical insights helpful for your research, please consider citing our paper:
```bibtex

```