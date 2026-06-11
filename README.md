# BdSL-NMM
## Bangladeshi Sign Language with Non-Manual Markers

The first BdSL dataset with annotated non-manual 
expression markers (NMM). Contains 4,170 samples 
across 62 sign word classes and 5 expression classes 
from 7 native signers, evaluated under strict 
leave-one-signer-out cross-validation.

## Dataset Download

The BdSL-NMM dataset (1.2 GB) is hosted separately.

**Download:** N/A

After downloading, extract and place at:
dataset/multimodal_6signers_clean/

The folder must contain 4,170 .npz files and 
label_mapping_clean.json

## Dataset Statistics
- Word classes: 62
- Expression classes: 5 (neutral, happy, sad, 
  negation, question)
- Total samples: 4,170
- Signers: 7
- Test signer: S7 (601 samples, LOSO held-out)
- Train+val signers: S1-S6 (3,569 samples)
- Mean samples per word class: 67.26
- Sequence length: 60 frames (padded/cropped)
- Input: MediaPipe Holistic skeletal landmarks

See data/data_analysis_report.txt for full 
per-class and per-signer statistics.

## Installation

pip install -r requirements.txt

GPU training requires CUDA-compatible PyTorch.
CPU training works automatically but is much slower.

## Training

All commands run from this folder.
CUDA is used automatically if available.

### SingleStreamBaseline (LOSO)
python train_model.py \
  --normalized_dir dataset/multimodal_6signers_clean \
  --epochs 50 --batch_size 16 \
  --learning_rate 0.0001 --dropout 0.2 \
  --checkpoint_name my_baseline \
  --model_type baseline

### BdSL-SPOTER (LOSO)
python train_model.py \
  --normalized_dir dataset/multimodal_6signers_clean \
  --epochs 50 --batch_size 16 \
  --learning_rate 0.0001 --dropout 0.2 \
  --checkpoint_name my_spoter \
  --model_type spoter

### SignNetV2 word-only (LOSO)
python train_model.py \
  --normalized_dir dataset/multimodal_6signers_clean \
  --epochs 50 --batch_size 16 \
  --learning_rate 0.0001 --dropout 0.2 \
  --num_emotions 0 \
  --checkpoint_name my_signetv2_words \
  --model_type signetv2

### SignNetV2 multitask word+expression (LOSO)
python train_model.py \
  --normalized_dir dataset/multimodal_6signers_clean \
  --epochs 50 --batch_size 16 \
  --learning_rate 0.0001 --dropout 0.2 \
  --emotion_loss_weight 0.5 --num_emotions 5 \
  --checkpoint_name my_signetv2_multitask \
  --model_type signetv2

## Evaluation

### Evaluate on S7 test signer (LOSO)
python eval_s11_final.py \
  --model_type signetv2 \
  --checkpoint_dir checkpoints/model_signetv2_clean

python eval_s11_final.py \
  --model_type baseline \
  --checkpoint_dir checkpoints/model_baseline_clean

python eval_s11_final.py \
  --model_type spoter \
  --checkpoint_dir checkpoints/model_spoter_clean

## Pre-trained Checkpoints

Four pre-trained models are included in checkpoints/:
- model_baseline_clean: SingleStreamBaseline
- model_spoter_clean: BdSL-SPOTER reproduced
- model_signetv2_wordonly: SignNetV2 word-only
- model_signetv2_clean: SignNetV2 multitask (best)

## Results
See RESULTS.md for full canonical results table.

## Architecture
SignNetV2 uses four stream-specific Transformer 
encoders (body pose, left hand, right hand, face), 
cross-stream attention fusion, hierarchical temporal 
encoding, and a multi-task classification head.
